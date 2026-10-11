#pragma once

// MoE expert streaming: the routed expert weights stay on disk and are read on demand into a
// bounded per-layer cache of expert slots. Ported from mihailescu2m/llama.cpp (MIT), commit d1762fc,
// without its cache resizing (phase switching), GPU slot resolution, PLE row reads and hash routers.

#include "llama-mmap.h"

#include "ggml-cpp.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// LLAMA_MOE_STREAM_PARTITION: give each (token, expert) pair to exactly one prefill wave instead of
// running every wave over every pair and masking the rest. On by default; 0 disables. Read by the
// stream and by build_moe_ffn, which must agree: the graph sizes the tensors the planner fills.
bool moe_stream_partition();

// Streamed layers do not load their ffn_*_exps tensors. Each weight gets a device cache tensor of
// n_slots expert slabs instead, filled from the GGUF file by a remapping custom op that runs on the
// CPU right after the router top-k. The op only changes which cache slot an expert id resolves to,
// never which experts the router picked, so streaming changes latency, not outputs.
//
// Misses are read by a pool of I/O threads while the op waits; eviction is by decaying route
// hotness with an LRU tiebreak.
//
// note: only one context may decode a streamed model at a time - one context can evict slots
// referenced by the other's in-flight graph.

// expert slab read latency histogram: upper bounds in microseconds, plus one bucket above the last
static const int64_t MOE_STREAM_READ_BUCKET_US[] = { 100, 250, 500, 1000, 2000, 4000, 8000 };
static const int     MOE_STREAM_READ_BUCKETS     = 8;

struct llama_moe_stream;
struct llama_moe_stream_layer;

enum llama_moe_stream_slot_state : uint8_t {
    LLAMA_MOE_STREAM_SLOT_EMPTY    = 0,
    LLAMA_MOE_STREAM_SLOT_LOADING  = 1, // reserved, load queued or in flight
    LLAMA_MOE_STREAM_SLOT_RESIDENT = 2,
};

// one streamed weight tensor (gate, up, down or fused gate_up) of one layer
struct llama_moe_stream_weight {
    ggml_tensor * cache = nullptr; // cache tensor {ne0, ne1, n_slots}

    uint16_t file_idx  = 0; // GGUF split file index
    size_t   offs      = 0; // file offset of the full exps tensor data
    size_t   nb_expert = 0; // bytes per expert slab
};

// userdata of the remap op with one-layer-ahead prefetch: the op remaps THIS layer as usual and
// also predicts the NEXT layer's routing from this layer's router input, issuing those reads early.
// The prediction skips the attention and FFN between the layers, so it is a lower bound.
struct llama_moe_stream_lookahead {
    llama_moe_stream_layer * sl       = nullptr; // this layer, remapped as usual
    llama_moe_stream_layer * sl_next  = nullptr; // layer to prefetch into, null = plain remap
    uint32_t                 top_k    = 0;       // how many predicted experts to fetch
    ggml_tensor *            bias_src = nullptr; // next layer's exp_probs_b, may be null
    bool                     bias_read = false;
    std::vector<float>       bias;               // host copy of bias_src, filled on first use
    std::vector<float>       score;              // scratch [n_expert]
    std::vector<int32_t>     selected;           // bounded, unique prefetch candidates
};

// userdata of one wave's custom ops (multi-pass prefill): identifies which pass this is
struct llama_moe_stream_wave {
    llama_moe_stream_layer * sl   = nullptr;
    int32_t                  wave = -1;
};

// per-layer streaming state - also the userdata of the remap op
struct llama_moe_stream_layer {
    llama_moe_stream * mgr = nullptr;

    int32_t  il       = -1;
    uint32_t n_expert = 0;
    uint32_t n_slots  = 0;

    std::vector<llama_moe_stream_weight> weights; // 2 (fused gate_up + down) or 3 entries

    // residency state, guarded by mgr->mtx
    std::vector<int32_t>                 slot_expert;   // [n_slots] expert id or -1
    std::vector<uint8_t>                 slot_state;    // [n_slots] llama_moe_stream_slot_state
    std::vector<uint8_t>                 slot_pending;  // [n_slots] slabs still in flight for this slot
    std::vector<uint64_t>                slot_gen;      // [n_slots] reservation generation
    std::vector<int64_t>                 slot_last_use; // [n_slots] LRU stamps
    std::unordered_map<int32_t, int32_t> expert_slot;   // RESIDENT and LOADING entries

    std::vector<uint32_t> route_hotness; // [n_expert] decayed selection counts, for eviction
    std::vector<uint8_t>  seen;          // [n_expert] for cold-miss attribution

    // prefetch accounting: who loaded each slot and whether a decode remap used it before eviction
    std::vector<uint8_t> slot_src;    // [n_slots] llama_moe_stream_src
    std::vector<uint8_t> slot_used;   // [n_slots]
    std::vector<int64_t> evicted_at;  // [n_expert] n_remap at the expert's last eviction, 0 = never
    int64_t              n_remap = 0; // decode remap calls of this layer
    int64_t use_counter = 0;

    // scratch for the op callbacks
    std::vector<int32_t> uniq;
    std::vector<uint8_t> touched;
    std::vector<uint8_t> keep;         // [n_slots] slots the current call must not evict
    std::vector<int32_t> demand_slots; // slots the current call waits on
    std::vector<uint8_t> defer;        // two-phase remap: pairs the first half left to the second
    float miss_ema  = 0.0f;  // share of recent small-batch remaps that had a miss
    bool  two_phase = false; // graph builds use the two-phase remap for this layer (hysteresis on miss_ema)

    // wave plan for multi-pass prefill (guarded by mgr->mtx): the touched experts are split into
    // plan_n_waves passes of at most plan_capacity experts each, run one pass at a time
    uint32_t plan_capacity  = 0;  // experts per wave, set at graph build
    uint32_t plan_n_waves   = 0;  // waves of the current call
    int32_t  plan_next_wave = -1; // wave expected to run next (ordering guard)
    std::vector<uint8_t> expert_wave; // [n_expert] wave of each touched expert, 0xff = untouched
    std::vector<int32_t> plan_pool;   // resident slots the masked-out pairs of this wave park on
    std::vector<int32_t> pool_used;   // scratch: pool slots already used in the current token row

    // pair partitioning (LLAMA_MOE_STREAM_PARTITION): every (token, expert) pair belongs to one wave,
    // so a wave runs the expert GEMMs over a dense pair list instead of over every pair with a mask.
    // plan_pair_chunk is fixed at graph build because ggml shapes are static; short waves pad.
    uint32_t plan_pair_chunk = 0;                 // static per-wave bound, set at graph build
    uint32_t plan_waves_want = 0;                 // wave count the graph built, 0 = not partitioned
    std::vector<uint32_t> wave_first, wave_count; // [n_waves] this wave's slice of uniq
    std::vector<std::vector<int32_t>> plan_pair;  // [n_waves][<= plan_pair_chunk] flat idx t*n_ids+k
    std::vector<int32_t>              pair_count; // [n_expert] pairs per expert, for wave balancing

    std::vector<std::unique_ptr<llama_moe_stream_wave>> wave_ud; // stable per-wave op userdata

    // stable userdata for wave w (grows lazily); graph build time only
    llama_moe_stream_wave * wave_userdata(int32_t wave, uint32_t capacity);

    // whether the exps tensors passed to build_moe_ffn are this layer's cache tensors
    bool matches(const ggml_tensor * gate, const ggml_tensor * up,
                 const ggml_tensor * down, const ggml_tensor * gate_up) const;

    // one-layer-ahead prefetch (LLAMA_MOE_STREAM_LOOKAHEAD), set by the model at load time
    ggml_tensor *                     la_gate_inp = nullptr; // next layer's ffn_gate_inp
    std::unique_ptr<llama_moe_stream_lookahead> la;

    // same-layer hint (LLAMA_MOE_STREAM_HINT): this layer's router run on the layer input before attention, by a host
    // op at the start of the layer. Its top experts are read while the GPU computes the attention; the remap op that
    // decides stays the one after the real router, so a wrong hint costs reads, never results.
    std::vector<int32_t> hint_pred;          // experts the last hint predicted for this layer
    std::vector<uint8_t> hint_mark;          // [n_expert] 1 = in hint_pred
    bool                 hint_valid = false; // hint_pred belongs to the call the next remap serves
    std::vector<float>   hint_scratch;

    // F32 [n_expert, hint_max_tokens] outside the graph's compute buffer: the GPU copies the hint logits here, so the
    // async hint op reads them after the GPU has gone on and reused the graph memory (null when hints are off)
    ggml_tensor * hint_store = nullptr;
};

// who started a slot's load
enum llama_moe_stream_src : uint8_t {
    LLAMA_MOE_STREAM_SRC_DEMAND    = 0,
    LLAMA_MOE_STREAM_SRC_LOOKAHEAD = 1,
    LLAMA_MOE_STREAM_SRC_HINT      = 2,
    LLAMA_MOE_STREAM_SRC_WAVE      = 3,
    LLAMA_MOE_STREAM_SRC_COUNT     = 4,
};

// one queued slab read
struct llama_moe_stream_work {
    llama_moe_stream_layer * sl = nullptr;

    int32_t  expert = -1;
    int32_t  slot   = -1;
    int32_t  widx   = -1;    // which weight slab of the expert; one work item per slab
    uint64_t gen    = 0;     // stale unless it matches slot_gen[slot]
    bool     demand = false; // in-flight work can be promoted after leaving the speculative queue
};

// cache tensors of one streamed layer and one buffer type, with their backing buffer
struct llama_moe_stream_buffer {
    int32_t                    il   = -1;
    ggml_backend_buffer_type_t buft = nullptr;
    ggml_context_ptr           ctx;
    ggml_backend_buffer_ptr    buf;
};

struct llama_moe_stream {
    uint32_t n_slots      = 0; // expert cache slots per streamed layer
    int32_t  n_io_threads = 0;

    std::vector<std::unique_ptr<llama_moe_stream_layer>> layers; // [n_layer], null = not streamed

    llama_moe_stream(uint32_t n_layer, uint32_t n_slots, int32_t n_io_threads, bool direct);
    ~llama_moe_stream();

    llama_moe_stream_layer * layer(int32_t il) const {
        return il >= 0 && (size_t) il < layers.size() ? layers[il].get() : nullptr;
    }

    // registers a streamed weight of layer il and returns its cache tensor
    ggml_tensor * create_cache_tensor(
            int32_t il, ggml_backend_buffer_type_t buft, const ggml_tensor * meta,
            uint16_t file_idx, size_t offs);

    // allocate the cache buffers (after all create_cache_tensor calls)
    void alloc_bufs(bool no_alloc);

    // reopen the GGUF files for streaming reads
    void open_files(const std::vector<std::string> & paths);

    size_t size_bufs() const;

    void print_stats() const;

    // LLAMA_MOE_STREAM_NO_ZEROCOPY: stage reads through a bounce buffer even when the cache is
    // host-visible (measured no better than reading straight into the slot)
    bool no_zerocopy = false;

    // same-layer hint (LLAMA_MOE_STREAM_HINT=K): top-K experts per token row, 0 = off;
    // LLAMA_MOE_STREAM_HINT_READ=0 only measures the prediction, 1 (default) also starts the reads
    uint32_t hint_k    = 0;
    bool     hint_read = true;
    uint32_t hint_max_tokens = 16; // graphs with more tokens (prefill) get no hint

    bool use_direct_io = false; // F_NOCACHE / O_DIRECT reads, no page cache

    llama_files files; // privately reopened GGUF files, same indices as the loader's

    std::vector<llama_moe_stream_buffer> cache_bufs; // one per streamed layer and buffer type

    ggml_context_ptr        hint_ctx; // the layers' hint_store tensors
    ggml_backend_buffer_ptr hint_buf;

    size_t  max_nb_expert      = 0;
    int64_t hot_decay_interval = 0; // remap calls between route-hotness halvings (0 = no decay)

    // whether partition-path padding may use slot -1 (see pad_skip_ok), -1 = not decided yet
    int8_t pad_skip = -1;
    bool pad_skip_ok();

    // load pool (queues and all layer residency state guarded by mtx)
    mutable std::mutex      mtx;
    std::condition_variable cv_work; // queued work or shutdown
    std::condition_variable cv_done; // a load committed or failed

    std::deque<llama_moe_stream_work> q_demand;

    // speculative reads (lookahead) in their own queue, drained only when no demand read waits:
    // with one FIFO a wide prefetch delays the read a layer is blocked on
    std::deque<llama_moe_stream_work> q_spec;

    // cap on the speculative backlog: a prefetch still queued when its layer arrives has only
    // reserved a slot and spent bandwidth
    size_t q_spec_max = 0;

    size_t demand_inflight = 0;
    std::vector<llama_moe_stream_work> reads_inflight; // one entry per worker, guarded by mtx

    std::vector<std::thread> workers;
    bool workers_started = false;
    bool shutting_down   = false;
    bool load_failed     = false;
    bool no_alloc        = false;
    size_t workers_active = 0;

    bool debug = false;

    struct llama_moe_stream_stats {
        int64_t n_calls     = 0; // remap invocations
        int64_t n_hit       = 0; // touched experts already resident or loading
        int64_t n_miss      = 0; // demand loads issued
        int64_t n_miss_cold = 0; // first-ever touch of an expert
        int64_t t_stall_us  = 0; // wait time in miss handling

        int64_t n_demand_self_evict = 0; // a miss evicted an expert touched by the same call

        int64_t n_wave_calls     = 0; // wave op invocations (>= n_calls under multi-pass prefill)
        int64_t n_waves_run      = 0; // non-empty waves
        int64_t n_preload_issued = 0; // next-wave and lookahead loads started early
        int64_t n_preload_ready  = 0; // wave experts already resident from the previous preload

        // a demand hit on a slot still loading means the prefetch picked the right expert too late
        int64_t n_hit_loading = 0;
        int64_t n_hit_ready   = 0;
        int64_t t_stall_wave_us = 0; // wait time in wave miss handling

        // wall time inside the streaming custom ops, stall included; the GPU is idle meanwhile
        int64_t t_wave_op_us  = 0;
        int64_t t_remap_op_us = 0;

        int64_t pair_over_max  = 0; // worst (max wave load / mean - 1), percent
        int64_t chunk_util_max = 0; // worst wave load as a percent of plan_pair_chunk (100 = abort)
        int64_t n_pair_splits  = 0; // hot experts split across waves to fit the chunk

        // per-miss anatomy: SSD read vs upload, slabs and bytes moved
        int64_t t_io_read_us   = 0;
        int64_t t_io_upload_us = 0;
        int64_t n_slabs_read   = 0;
        int64_t n_bytes_read   = 0;

        int64_t n_spec_dispatch    = 0; // speculative reads started
        int64_t n_spec_with_demand = 0; // of those, started while demand work was pending

        // same-layer hint: what it predicted against what the router then picked
        int64_t n_hint_calls    = 0; // hint ops run
        int64_t n_hint_issued   = 0; // expert loads the hints started
        int64_t n_hint_touched  = 0; // experts the router picked on a hinted call
        int64_t n_hint_covered  = 0; // of those, predicted by the hint
        int64_t n_hint_miss     = 0; // misses (not resident, not loading) on a hinted call ...
        int64_t n_hint_miss_cov = 0; // ... the hint had predicted (a read it could have started early)
        int64_t n_hint_late     = 0; // router picks still loading from a hint when the remap ran

        // prefetched loads used by a decode remap before eviction, and evicted unused; misses on experts evicted
        // within the last 256 remaps of their layer
        int64_t n_pf_used  [LLAMA_MOE_STREAM_SRC_COUNT] = {0};
        int64_t n_pf_wasted[LLAMA_MOE_STREAM_SRC_COUNT] = {0};
        int64_t n_miss_recent = 0;
        int64_t n_pf_cancelled[LLAMA_MOE_STREAM_SRC_COUNT] = {0}; // dropped from the queue before any read started

        int64_t n_read_bucket[MOE_STREAM_READ_BUCKETS] = {0};
    };

    llama_moe_stream_stats stats;

    // LLAMA_MOE_STREAM_TRACE=<path>: one text line per routing call, for offline cache-policy simulation:
    // "D <layer> <n> <experts...>" for a decode remap, "P <layer> <n> <experts...>" for a prefill ubatch
    FILE * trace = nullptr;
    void trace_locked(char kind, int32_t il, const std::vector<int32_t> & experts);

    // periodic delta dump (LLAMA_MOE_STREAM_STATS_MS=<ms>), at WARN so llama-server shows it
    int64_t                stats_dump_us   = 0; // interval, 0 = disabled
    int64_t                stats_t_last_us = 0;
    llama_moe_stream_stats stats_prev;
    void maybe_dump_stats_locked();

    // internals
    void start_workers_locked();
    void worker_loop(size_t worker_id);
    int32_t pick_victim_locked(llama_moe_stream_layer & sl, const uint8_t * keep) const;
    void reserve_slot_locked(llama_moe_stream_layer & sl, int32_t expert, int32_t slot,
                             uint8_t src = LLAMA_MOE_STREAM_SRC_DEMAND);
    void promote_slot_locked(llama_moe_stream_layer & sl, int32_t slot);

    // multi-pass prefill helpers (called by the wave ops, all under mtx)
    void plan_waves_locked(llama_moe_stream_layer & sl, const int32_t * ids, int64_t n);
    void plan_pairs_locked(llama_moe_stream_layer & sl, const int32_t * ids, int64_t n);
    void stage_wave_locked(std::unique_lock<std::mutex> & lk, llama_moe_stream_layer & sl, int32_t w, uint32_t n_ids);
    void emit_wave_slots(llama_moe_stream_layer & sl, const int32_t * ids, int32_t * out, int32_t w, uint32_t n_ids, int64_t n_tok);
    void emit_wave_pairs(llama_moe_stream_layer & sl, const int32_t * ids, int32_t * out, int32_t w, uint32_t n_ids, int64_t n_pairs, int64_t chunk);
};

// callback of the remap op inserted by build_moe_ffn (decode and small ubatches)
void llama_moe_stream_remap(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata);

// same-layer hint op: src a holds this layer's router logits computed from the layer input before attention
// [n_expert, n_tokens]; starts reads for the top experts. Writes nothing; userdata is the llama_moe_stream_layer.
void llama_moe_stream_hint(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata);

// remap for THIS layer (src a) plus a lookahead prefetch for the next layer; src b holds the next
// layer's predicted router logits. userdata is a llama_moe_stream_lookahead.
void llama_moe_stream_remap_la(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b, int ith, int nth, void * userdata);
// two-phase remap for small batches (LLAMA_MOE_STREAM_TWO_PHASE): the first half maps the resident experts and starts
// the loads, the second half (a custom op over the router ids and an ordering token) waits and maps the rest
void llama_moe_stream_remap2a(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata);
void llama_moe_stream_remap2a_la(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b, int ith, int nth, void * userdata);
void llama_moe_stream_remap2b(ggml_tensor * dst, int ith, int nth, void * userdata);

// callbacks of the multi-pass prefill ops, used when a ubatch touches more experts than the cache
// holds; each src[0] is the contiguous selected ids
//   wave_ids:   make wave w's experts resident and emit slot ids (masked pairs park on a pool)
//   wave_mask:  1.0 for pairs of wave w, 0.0 otherwise
//   wave_pairs: same staging, but emit the index rows of wave w's dense pair list (partition path)
// Per-graph wave plan (pair chunk, wave count, wave capacity), written into the wave ops' op_params at graph build
// and read back by the ops. llama builds the next ubatch's graph while this one's host ops are still pending, so the
// per-layer plan_* fields can already belong to a graph of another shape when an op runs (the short last ubatch of a
// prompt; with 4096-token ubatches the planner then staged too few experts and a wave op aborted).
// Bytes 32..47 of op_params: after ggml_custom_op_params (24 bytes), before the Metal host-op value at 48.
void llama_moe_stream_set_plan(ggml_tensor * op, uint32_t pair_chunk, uint32_t waves_want, uint32_t capacity);

void llama_moe_stream_wave_ids  (ggml_tensor * dst, int ith, int nth, void * userdata);
void llama_moe_stream_wave_mask (ggml_tensor * dst, int ith, int nth, void * userdata);
void llama_moe_stream_wave_pairs(ggml_tensor * dst, int ith, int nth, void * userdata);

// index rows of the wave_pairs output, an I32 [chunk, LLAMA_MOE_PAIR_ROWS] tensor
enum llama_moe_pair_row {
    LLAMA_MOE_PAIR_TOK  = 0, // token index, to gather the GEMM input row
    LLAMA_MOE_PAIR_PAIR = 1, // flat pair index t*n_ids + k, to gather (weight_before_ffn) and scatter
    LLAMA_MOE_PAIR_SLOT = 2, // cache slot the GEMM indexes
    LLAMA_MOE_PAIR_EXP  = 3, // original expert id, for biases and per-expert scales
    LLAMA_MOE_PAIR_ROWS = 4,
};
