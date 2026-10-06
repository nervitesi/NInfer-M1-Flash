// MoE expert streaming. Ported from mihailescu2m/llama.cpp (MIT), commit d1762fc; see llama-moe-stream.h.

#include "llama-moe-stream.h"
#include "llama-moe-lookahead.h"

#include "llama-impl.h"

#include "ggml-backend.h"

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifdef _WIN32
#include <malloc.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

static const uint32_t MOE_STREAM_IO_THREADS_DEFAULT = 9;
static const uint32_t MOE_STREAM_IO_THREADS_MAX     = 18;

// route hotness halves every this many tokens; at 64 the counters stayed at 0-3 and eviction
// degenerated into plain LRU (measured upstream: 1024 gave +18% decode, miss 6.8% -> 5.4%)
static const int64_t MOE_STREAM_HOT_DECAY_TOKENS = 1024;

#if defined(__APPLE__)
static const char * const MOE_STREAM_DIRECT_NAME = "F_NOCACHE";
#else
static const char * const MOE_STREAM_DIRECT_NAME = "O_DIRECT";
#endif

// direct I/O alignment: a multiple of any logical block size
static const size_t MOE_STREAM_DIRECT_ALIGN = 4096;

// saturating increment - hotness counters must not wrap
static uint32_t sat_inc(uint32_t & c) {
    if (c < UINT32_MAX - 1) {
        c++;
    }
    return c;
}

static void * moe_aligned_alloc(size_t n) {
#ifdef _WIN32
    return _aligned_malloc(n, MOE_STREAM_DIRECT_ALIGN);
#else
    void * p = nullptr;
    if (posix_memalign(&p, MOE_STREAM_DIRECT_ALIGN, n) != 0) {
        p = nullptr;
    }
    return p;
#endif
}

static void moe_aligned_free(void * p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

// read len bytes at file offset offs into staging (thread-safe positional read); with direct I/O,
// staging needs room for len + 2*MOE_STREAM_DIRECT_ALIGN. returns a pointer to the len bytes within
// staging, or nullptr on failure
static const uint8_t * llama_moe_stream_pread(llama_file & file, uint8_t * staging, size_t len, size_t offs, bool direct) {
#ifdef _WIN32
    GGML_UNUSED(direct);
    static std::mutex io_mtx;
    std::lock_guard<std::mutex> lock(io_mtx);
    try {
        file.seek(offs, SEEK_SET);
        file.read_raw(staging, len);
        return staging;
    } catch (...) {
        return nullptr;
    }
#else
    const int fd = file.file_id();

    if (direct) {
        // direct I/O needs the offset, length and buffer block-aligned
        const size_t a     = MOE_STREAM_DIRECT_ALIGN;
        const size_t aoffs = offs & ~(a - 1);
        const size_t head  = offs - aoffs;
        const size_t total = ((head + len + a - 1)/a)*a;
        ssize_t r;
        do {
            r = pread(fd, staging, total, aoffs);
        } while (r < 0 && errno == EINTR);
        if (r < 0 || (size_t) r < head + len) {
            return nullptr;
        }
        return staging + head;
    }

    uint8_t * p    = staging;
    size_t    left = len;
    while (left > 0) {
        const ssize_t r = pread(fd, p, left, offs);
        if (r < 0) {
            if (errno == EINTR) {
                continue;
            }
            return nullptr;
        }
        if (r == 0) {
            return nullptr; // unexpected EOF
        }
        p    += r;
        offs += (size_t) r;
        left -= (size_t) r;
    }
    return staging;
#endif
}

bool llama_moe_stream_layer::matches(const ggml_tensor * gate, const ggml_tensor * up,
                                     const ggml_tensor * down, const ggml_tensor * gate_up) const {
    auto is_cache = [this](const ggml_tensor * t) {
        for (const auto & w : weights) {
            if (w.cache == t) {
                return true;
            }
        }
        return false;
    };

    size_t n = 0;
    for (const ggml_tensor * t : { gate, up, down, gate_up }) {
        if (t == nullptr) {
            continue;
        }
        if (!is_cache(t)) {
            return false;
        }
        n++;
    }

    return n > 0 && n == weights.size();
}

llama_moe_stream_wave * llama_moe_stream_layer::wave_userdata(int32_t wave, uint32_t capacity) {
    GGML_ASSERT(capacity >= 1 && capacity <= n_slots);
    plan_capacity = capacity;
    while ((size_t) wave >= wave_ud.size()) {
        auto ud = std::make_unique<llama_moe_stream_wave>();
        ud->sl   = this;
        ud->wave = (int32_t) wave_ud.size();
        wave_ud.push_back(std::move(ud));
    }
    return wave_ud[wave].get();
}

bool moe_stream_partition() {
    static const bool v = [] {
        const char * s = std::getenv("LLAMA_MOE_STREAM_PARTITION");
        return s == nullptr || atoi(s) != 0;
    }();
    return v;
}

llama_moe_stream::llama_moe_stream(uint32_t n_layer, uint32_t n_slots, int32_t n_io_threads, bool direct) : n_slots(n_slots) {
    layers.resize(n_layer);

    this->n_io_threads = n_io_threads <= 0 ? MOE_STREAM_IO_THREADS_DEFAULT : n_io_threads;
    this->n_io_threads = std::min<int32_t>(this->n_io_threads, MOE_STREAM_IO_THREADS_MAX);

    debug         = std::getenv("LLAMA_MOE_STREAM_DEBUG") != nullptr;
    use_direct_io = direct;

    if (const char * s = std::getenv("LLAMA_MOE_STREAM_STATS_MS")) {
        stats_dump_us = std::max<int64_t>(0, std::atoll(s))*1000;
    }

#ifdef GGML_USE_METAL
    // on Metal the remap and wave ops run as host ops inside the command stream instead of splitting
    // the graph at every streamed layer (LLAMA_MOE_STREAM_HOST_OPS=0 keeps the CPU splits)
    {
        const char * e = std::getenv("LLAMA_MOE_STREAM_HOST_OPS");
        if (!e || atoi(e) != 0) {
            ggml_backend_metal_register_host_op((void *) llama_moe_stream_remap);
            ggml_backend_metal_register_host_op((void *) llama_moe_stream_remap_la);
            ggml_backend_metal_register_host_op((void *) llama_moe_stream_wave_ids);
            ggml_backend_metal_register_host_op((void *) llama_moe_stream_wave_mask);
            ggml_backend_metal_register_host_op((void *) llama_moe_stream_wave_pairs);
        }
    }
#endif

    // keep every reader busy a few slabs deep, but no deeper
    q_spec_max = (size_t) this->n_io_threads * 8;
    if (const char * s = std::getenv("LLAMA_MOE_STREAM_SPEC_MAX")) {
        q_spec_max = (size_t) std::max(0, atoi(s));
    }
}

llama_moe_stream::~llama_moe_stream() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        shutting_down = true;
        q_demand.clear();
        q_spec.clear();
    }
    cv_work.notify_all();
    for (auto & w : workers) {
        w.join();
    }
}

ggml_tensor * llama_moe_stream::create_cache_tensor(
        int32_t il, ggml_backend_buffer_type_t buft, const ggml_tensor * meta,
        uint16_t file_idx, size_t offs) {
    GGML_ASSERT(il >= 0 && (size_t) il < layers.size());
    GGML_ASSERT(ggml_is_contiguous(meta));
    GGML_ASSERT(meta->ne[2] > 0 && meta->ne[3] == 1);

    const uint32_t n_expert  = meta->ne[2];
    const size_t   nb_expert = ggml_nbytes(meta) / n_expert;
    GGML_ASSERT(nb_expert * n_expert == ggml_nbytes(meta));
    GGML_ASSERT(n_slots > 0 && n_slots < n_expert);

    ggml_context * ctx = nullptr;
    for (auto & group : cache_bufs) {
        if (group.il == il && group.buft == buft) {
            ctx = group.ctx.get();
            break;
        }
    }
    if (ctx == nullptr) {
        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead()*5,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);
        if (ctx == nullptr) {
            throw std::runtime_error("failed to create ggml context for MoE expert streaming");
        }
        llama_moe_stream_buffer group;
        group.il   = il;
        group.buft = buft;
        group.ctx.reset(ctx);
        cache_bufs.emplace_back(std::move(group));
    }

    ggml_tensor * cache = ggml_new_tensor_3d(ctx, meta->type, meta->ne[0], meta->ne[1], n_slots);
    ggml_format_name(cache, "%s.stream_cache", meta->name);
    GGML_ASSERT(ggml_nbytes(cache) == nb_expert * n_slots);

    auto & sl = layers[il];
    if (!sl) {
        sl = std::make_unique<llama_moe_stream_layer>();
        sl->mgr      = this;
        sl->il       = il;
        sl->n_expert = n_expert;
        sl->n_slots  = n_slots;
        sl->slot_expert  .resize(n_slots, -1);
        sl->slot_state   .resize(n_slots, LLAMA_MOE_STREAM_SLOT_EMPTY);
        sl->slot_pending .resize(n_slots, 0);
        sl->slot_gen     .resize(n_slots, 0);
        sl->slot_last_use.resize(n_slots, 0);
        sl->route_hotness.resize(n_expert, 0);
        sl->seen         .resize(n_expert, 0);
        sl->keep         .resize(n_slots, 0);
    }
    GGML_ASSERT(sl->n_expert == n_expert);

    sl->weights.push_back({ cache, file_idx, offs, nb_expert });

    max_nb_expert = std::max(max_nb_expert, nb_expert);

    return cache;
}

void llama_moe_stream::alloc_bufs(bool no_alloc) {
    this->no_alloc = no_alloc;
    for (auto & group : cache_bufs) {
        ggml_backend_buffer_type_t buft = group.buft;
        ggml_context * ctx = group.ctx.get();
        if (ggml_get_first_tensor(ctx) == nullptr) {
            continue;
        }

        ggml_backend_buffer_t buf;
        if (no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                t->buffer = buf;
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        }
        if (buf == nullptr) {
            throw std::runtime_error(format("unable to allocate %s buffer for MoE expert streaming", ggml_backend_buft_name(buft)));
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        group.buf.reset(buf);

        LLAMA_LOG_DEBUG("%s: layer %d %12s expert cache size = %8.2f MiB (%u slots)\n",
                __func__, group.il, ggml_backend_buffer_name(buf),
                ggml_backend_buffer_get_size(buf) / 1024.0 / 1024.0, n_slots);
    }

    LLAMA_LOG_INFO("%s: expert cache size = %8.2f MiB (%u slots per layer, %zu layer buffers)\n",
            __func__, size_bufs() / 1024.0 / 1024.0, n_slots, cache_bufs.size());
}

void llama_moe_stream::open_files(const std::vector<std::string> & paths) {
    for (const auto & path : paths) {
        if (path.empty()) {
            throw std::runtime_error("MoE expert streaming requires a file-based model (not a stream/file descriptor)");
        }
    }

    auto open_all = [&](bool direct) {
        files.clear();
        for (const auto & path : paths) {
            files.emplace_back(new llama_file(path.c_str(), "rb", direct));
        }
    };

    open_all(use_direct_io);

    // fall back to buffered reads when direct I/O is unusable: the open did not honor it, or a probe
    // read fails. Done here, single-threaded, before any worker starts.
    if (use_direct_io) {
        bool ok = !files.empty() && files.front()->has_direct_io();
        if (ok) {
            uint8_t * probe = (uint8_t *) moe_aligned_alloc(MOE_STREAM_DIRECT_ALIGN);
            GGML_ASSERT(probe != nullptr);
            ok = llama_moe_stream_pread(*files.front(), probe, MOE_STREAM_DIRECT_ALIGN, 0, /*direct =*/ true) != nullptr;
            moe_aligned_free(probe);
        }
        if (!ok) {
            LLAMA_LOG_WARN("%s: %s not usable, falling back to buffered streaming reads\n",
                    __func__, MOE_STREAM_DIRECT_NAME);
            use_direct_io = false;
            open_all(false);
        }
    }

    if (use_direct_io) {
        LLAMA_LOG_WARN("%s: MoE expert streaming uses %s (page cache bypassed)\n",
                __func__, MOE_STREAM_DIRECT_NAME);
    }

    no_zerocopy = std::getenv("LLAMA_MOE_STREAM_NO_ZEROCOPY") != nullptr;

    for (const auto & sl : layers) {
        if (sl && !sl->weights.empty()) {
            const bool zc = !use_direct_io && !no_zerocopy &&
                    ggml_backend_tensor_get_host_ptr(sl->weights[0].cache) != nullptr;
            LLAMA_LOG_WARN("%s: MoE expert streaming reads %s\n",
                    __func__, zc ? "directly into the expert cache (no staging copy)"
                                 : "through a staging buffer");
            break;
        }
    }

    // one token drives one remap per streamed layer, so the decay interval in remap calls is
    // decay_tokens * n_streamed_layers. LLAMA_MOE_STREAM_HOT_DECAY sets it in tokens, 0 = never.
    int64_t n_streamed = 0;
    for (const auto & sl : layers) {
        n_streamed += sl != nullptr;
    }
    int64_t decay_tokens = MOE_STREAM_HOT_DECAY_TOKENS;
    if (const char * s = std::getenv("LLAMA_MOE_STREAM_HOT_DECAY")) {
        decay_tokens = std::max<int64_t>(0, std::atoll(s));
    }
    hot_decay_interval = decay_tokens * n_streamed;
}

// spawn the I/O thread pool on first use (under mtx)
void llama_moe_stream::start_workers_locked() {
    if (workers_started) {
        return;
    }
    workers_started = true;
    reads_inflight.resize(n_io_threads);
    workers.reserve(n_io_threads);
    for (int32_t i = 0; i < n_io_threads; i++) {
        workers.emplace_back([this, i]() { worker_loop((size_t) i); });
    }
}

// I/O worker: pops a reserved load, reads one expert slab from the GGUF file into the cache slot,
// and publishes the slot when its last slab lands (or flags load_failed); stale items are skipped
void llama_moe_stream::worker_loop(size_t worker_id) {
    // page-aligned staging with the head/tail slack direct I/O needs
    uint8_t * staging = (uint8_t *) moe_aligned_alloc(max_nb_expert + 2*MOE_STREAM_DIRECT_ALIGN);
    GGML_ASSERT(staging != nullptr);

    std::unique_lock<std::mutex> lk(mtx);
    while (true) {
        cv_work.wait(lk, [&] {
            return shutting_down || !q_demand.empty() || !q_spec.empty();
        });
        if (shutting_down) {
            break;
        }

        // demand first, always: a layer is blocked on those, nothing is blocked on a prefetch
        llama_moe_stream_work w;
        if (!q_demand.empty()) {
            w = q_demand.front();
            q_demand.pop_front();
            w.demand = true;
        } else {
            w = q_spec.front();
            q_spec.pop_front();
            w.demand = false;
        }

        auto & sl = *w.sl;
        // several workers may hold different slabs of the same slot at once; staleness is per slot
        if (w.slot < 0 || (uint32_t) w.slot >= sl.n_slots ||
            w.gen != sl.slot_gen[w.slot] ||
            sl.slot_state[w.slot] != LLAMA_MOE_STREAM_SLOT_LOADING ||
            sl.slot_expert[w.slot] != w.expert ||
            w.widx < 0 || (size_t) w.widx >= sl.weights.size()) {
            cv_done.notify_all();
            continue; // stale item
        }

        workers_active++;
        reads_inflight[worker_id] = w;
        if (w.demand) {
            demand_inflight++;
        } else {
            stats.n_spec_dispatch++;
            stats.n_spec_with_demand += demand_inflight > 0;
        }

        lk.unlock();

        // read straight into the cache slot when the backend hands out a host pointer: on unified
        // memory the slot is host memory, so staging and uploading is a pure extra copy. Direct I/O
        // keeps staging for its block-aligned head/tail.
        const auto & wt = sl.weights[w.widx];
        uint8_t * dst = nullptr;
        if (!use_direct_io && !no_zerocopy) {
            auto * host = (uint8_t *) ggml_backend_tensor_get_host_ptr(wt.cache);
            dst = host ? host + (size_t) w.slot*wt.nb_expert : nullptr;
        }

        const int64_t t0 = ggml_time_us();
        const uint8_t * data = llama_moe_stream_pread(*files[wt.file_idx], dst ? dst : staging,
                wt.nb_expert, wt.offs + (size_t) w.expert*wt.nb_expert, use_direct_io);
        const int64_t t1 = ggml_time_us();
        const bool ok = data != nullptr;
        if (ok && dst == nullptr) {
            ggml_backend_tensor_set(wt.cache, data, (size_t) w.slot*wt.nb_expert, wt.nb_expert);
        }
        const int64_t t2 = ggml_time_us();

        lk.lock();

        GGML_ASSERT(workers_active > 0);
        workers_active--;
        // read the live flag: a speculative read can become demand while the I/O runs
        if (reads_inflight[worker_id].demand) {
            GGML_ASSERT(demand_inflight > 0);
            demand_inflight--;
        }
        reads_inflight[worker_id] = {};

        stats.t_io_read_us   += t1 - t0;
        stats.t_io_upload_us += ok ? t2 - t1 : 0;
        stats.n_slabs_read   += 1;
        stats.n_bytes_read   += ok ? wt.nb_expert : 0;

        {
            int b = 0;
            while (b < MOE_STREAM_READ_BUCKETS - 1 && (t1 - t0) >= MOE_STREAM_READ_BUCKET_US[b]) {
                b++;
            }
            stats.n_read_bucket[b]++;
        }

        if (!ok) {
            load_failed = true;
            sl.slot_pending[w.slot] = 0;
        } else if (w.gen == sl.slot_gen[w.slot] && sl.slot_pending[w.slot] > 0) {
            // the LAST slab to land publishes the slot, so no consumer sees a half-filled expert
            if (--sl.slot_pending[w.slot] == 0) {
                sl.slot_state[w.slot] = LLAMA_MOE_STREAM_SLOT_RESIDENT;
            }
        }
        cv_done.notify_all();
    }
    lk.unlock();

    moe_aligned_free(staging);
}

// least valuable evictable slot: empty first, then coldest resident (min route hotness, oldest use
// as tiebreak); LOADING and keep slots are never candidates. returns -1 when none exists
int32_t llama_moe_stream::pick_victim_locked(llama_moe_stream_layer & sl, const uint8_t * keep) const {
    int32_t v = -1;

    for (uint32_t s = 0; s < sl.n_slots; s++) {
        if ((keep && keep[s]) || sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_LOADING) {
            continue;
        }
        if (sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_EMPTY) {
            return s;
        }
        if (v < 0) {
            v = s;
            continue;
        }
        const uint32_t hs = sl.route_hotness[sl.slot_expert[s]];
        const uint32_t hv = sl.route_hotness[sl.slot_expert[v]];
        if (hs < hv || (hs == hv && sl.slot_last_use[s] < sl.slot_last_use[v])) {
            v = s;
        }
    }

    return v;
}

// bind expert -> slot and mark it LOADING: evict the prior occupant, bump slot_gen (so in-flight
// loads for the old occupant are stale) and update the expert_slot index
void llama_moe_stream::reserve_slot_locked(llama_moe_stream_layer & sl, int32_t expert, int32_t slot) {
    if (sl.slot_expert[slot] >= 0) {
        if (debug) {
            LLAMA_LOG_DEBUG("%s: layer %d: evict expert %d from slot %d\n", __func__, sl.il, sl.slot_expert[slot], slot);
        }
        sl.expert_slot.erase(sl.slot_expert[slot]);
    }

    sl.slot_expert[slot] = expert;
    sl.slot_state[slot]  = LLAMA_MOE_STREAM_SLOT_LOADING;
    sl.slot_gen[slot]++;
    sl.slot_last_use[slot] = ++sl.use_counter;
    sl.expert_slot[expert] = slot;
    sl.seen[expert] = 1;
}

// a speculative hit has one work item per slab already: move the queued items to the demand queue
// and mark the in-flight ones as required (re-enqueueing would publish the slot too early)
void llama_moe_stream::promote_slot_locked(llama_moe_stream_layer & sl, int32_t slot) {
    for (auto & w : reads_inflight) {
        if (w.sl == &sl && w.slot == slot && w.gen == sl.slot_gen[slot] && !w.demand) {
            w.demand = true;
            demand_inflight++;
        }
    }
    bool moved = false;
    for (auto it = q_spec.begin(); it != q_spec.end();) {
        if (it->sl == &sl && it->slot == slot && it->gen == sl.slot_gen[slot]) {
            q_demand.push_back(*it);
            it = q_spec.erase(it);
            moved = true;
        } else {
            ++it;
        }
    }
    if (moved) {
        cv_work.notify_all();
    }
}

size_t llama_moe_stream::size_bufs() const {
    size_t size = 0;
    for (const auto & group : cache_bufs) {
        if (group.buf) {
            size += ggml_backend_buffer_get_size(group.buf.get());
        }
    }
    return size;
}

void llama_moe_stream::maybe_dump_stats_locked() {
    if (stats_dump_us <= 0) {
        return;
    }

    const int64_t now = ggml_time_us();
    if (stats_t_last_us == 0) {
        stats_t_last_us = now;
        stats_prev      = stats;
        return;
    }
    const int64_t dt = now - stats_t_last_us;
    if (dt < stats_dump_us) {
        return;
    }

    const int64_t d_calls   = stats.n_calls          - stats_prev.n_calls;
    const int64_t d_hit     = stats.n_hit            - stats_prev.n_hit;
    const int64_t d_miss    = stats.n_miss           - stats_prev.n_miss;
    const int64_t d_waves   = stats.n_waves_run      - stats_prev.n_waves_run;
    const int64_t d_pre_i   = stats.n_preload_issued - stats_prev.n_preload_issued;
    const int64_t d_pre_r   = stats.n_preload_ready  - stats_prev.n_preload_ready;
    const int64_t d_hit_l   = stats.n_hit_loading    - stats_prev.n_hit_loading;
    const int64_t d_hit_r   = stats.n_hit_ready      - stats_prev.n_hit_ready;
    const int64_t d_stall   = (stats.t_stall_us      - stats_prev.t_stall_us) +
                              (stats.t_stall_wave_us - stats_prev.t_stall_wave_us);
    const int64_t d_op      = (stats.t_wave_op_us    - stats_prev.t_wave_op_us) +
                              (stats.t_remap_op_us   - stats_prev.t_remap_op_us);
    const int64_t d_cpu     = d_op > d_stall ? d_op - d_stall : 0;
    const int64_t d_touched = d_hit + d_miss;

    // only windows that did work
    if (d_calls > 0 || d_waves > 0) {
        LLAMA_LOG_WARN("%s: moe stream: %6.2f s | remaps %5" PRId64 " waves %4" PRId64
                       " | miss %5" PRId64 "/%-6" PRId64 " (%5.1f%%) | hit rdy/late %5" PRId64 "/%-5" PRId64
                       " | preload %4" PRId64 "->%-4" PRId64
                       " | stall %5.1f%%  cpu-op %5.1f%%  rest(gpu) %5.1f%%\n",
                __func__, dt/1e6, d_calls, d_waves, d_miss, d_touched,
                d_touched > 0 ? 100.0*d_miss/d_touched : 0.0,
                d_hit_r, d_hit_l,
                d_pre_i, d_pre_r,
                dt > 0 ? 100.0*d_stall/dt      : 0.0,
                dt > 0 ? 100.0*d_cpu/dt        : 0.0,
                dt > 0 ? 100.0*(dt - d_op)/dt  : 0.0);

        const int64_t d_self_evict = stats.n_demand_self_evict - stats_prev.n_demand_self_evict;
        const int64_t d_spec = stats.n_spec_dispatch - stats_prev.n_spec_dispatch;
        if (d_spec > 0) {
            LLAMA_LOG_WARN("%s: moe stream: prefetch dispatch %" PRId64 " while-demand %" PRId64
                           " | expert read MiB %.2f\n",
                    __func__, d_spec, stats.n_spec_with_demand - stats_prev.n_spec_with_demand,
                    (stats.n_bytes_read - stats_prev.n_bytes_read)/1048576.0);
        }
        if (d_self_evict > 0) {
            LLAMA_LOG_WARN("%s: moe stream: demand self-evictions = %" PRId64 " in this interval\n", __func__, d_self_evict);
        }

        const int64_t d_rd = stats.t_io_read_us   - stats_prev.t_io_read_us;
        const int64_t d_up = stats.t_io_upload_us - stats_prev.t_io_upload_us;
        const int64_t d_sl = stats.n_slabs_read   - stats_prev.n_slabs_read;
        if (d_sl > 0) {
            LLAMA_LOG_WARN("%s: moe stream: slabs %5" PRId64 " | read %7.2f ms (%5.3f ms/slab) | "
                           "upload %7.2f ms (%5.3f ms/slab) | read %4.1f%% of miss\n",
                    __func__, d_sl, d_rd/1000.0, d_rd/1000.0/d_sl, d_up/1000.0, d_up/1000.0/d_sl,
                    (d_rd + d_up) > 0 ? 100.0*d_rd/(d_rd + d_up) : 0.0);

            // reads faster than the first bound did not come from the drive: page cache hits
            char hist[256];
            int  off = 0;
            int64_t fast = 0;
            for (int b = 0; b < MOE_STREAM_READ_BUCKETS; b++) {
                const int64_t d = stats.n_read_bucket[b] - stats_prev.n_read_bucket[b];
                if (b == 0) {
                    fast = d;
                }
                off += snprintf(hist + off, sizeof(hist) - off, "%s%" PRId64, b ? "/" : "", d);
            }
            LLAMA_LOG_WARN("%s: moe stream: read us <100/<250/<500/<1k/<2k/<4k/<8k/more = %s | "
                           "page-cache L2 %4.1f%%\n",
                    __func__, hist, d_sl > 0 ? 100.0*fast/d_sl : 0.0);
        }

        if (stats.chunk_util_max > 0) {
            LLAMA_LOG_WARN("%s: moe stream: chunk utilisation worst = %" PRId64 "%% (100%% = abort)\n",
                    __func__, stats.chunk_util_max);
        }
        if (stats.pair_over_max > 0) {
            LLAMA_LOG_WARN("%s: moe stream: pair imbalance worst = +%" PRId64 "%% over mean\n",
                    __func__, stats.pair_over_max);
        }
    }

    stats_t_last_us = now;
    stats_prev      = stats;
}

void llama_moe_stream::print_stats() const {
    std::lock_guard<std::mutex> lock(mtx);

    const int64_t n_touched = stats.n_hit + stats.n_miss;
    LLAMA_LOG_WARN("%s: moe stream: remap calls = %" PRId64 ", expert hits = %" PRId64 ", misses = %" PRId64 " (%" PRId64 " cold), hit rate = %.2f%%\n",
            __func__, stats.n_calls, stats.n_hit, stats.n_miss, stats.n_miss_cold,
            n_touched > 0 ? 100.0*stats.n_hit/n_touched : 0.0);
    LLAMA_LOG_WARN("%s: moe stream: load stall = %.2f ms total (%.3f ms per remap call)\n",
            __func__, stats.t_stall_us/1000.0, stats.n_calls > 0 ? stats.t_stall_us/1000.0/stats.n_calls : 0.0);
    LLAMA_LOG_WARN("%s: moe stream: demand self-evictions = %" PRId64 "\n", __func__, stats.n_demand_self_evict);
    LLAMA_LOG_WARN("%s: moe stream: prefetch dispatch %" PRId64 " while-demand %" PRId64
                   " | expert read MiB %.2f\n",
            __func__, stats.n_spec_dispatch, stats.n_spec_with_demand, stats.n_bytes_read/1048576.0);
    if (stats.n_wave_calls > 0) {
        LLAMA_LOG_WARN("%s: moe stream: waves = %" PRId64 " (%" PRId64 " non-empty), preloads issued = %" PRId64 " (ready on arrival = %" PRId64 "), wave stall = %.2f ms\n",
                __func__, stats.n_wave_calls, stats.n_waves_run, stats.n_preload_issued, stats.n_preload_ready, stats.t_stall_wave_us/1000.0);
    }
    if (stats.n_slabs_read > 0) {
        for (int b = 0; b < MOE_STREAM_READ_BUCKETS; b++) {
            const int64_t lo = b ? MOE_STREAM_READ_BUCKET_US[b-1] : 0;
            if (b == MOE_STREAM_READ_BUCKETS - 1) {
                LLAMA_LOG_WARN("%s: moe stream: read %6" PRId64 " us +      : %8" PRId64 " slabs (%4.1f%%)\n",
                        __func__, lo, stats.n_read_bucket[b], 100.0*stats.n_read_bucket[b]/stats.n_slabs_read);
            } else {
                LLAMA_LOG_WARN("%s: moe stream: read %6" PRId64 " - %6" PRId64 " us: %8" PRId64 " slabs (%4.1f%%)\n",
                        __func__, lo, MOE_STREAM_READ_BUCKET_US[b], stats.n_read_bucket[b],
                        100.0*stats.n_read_bucket[b]/stats.n_slabs_read);
            }
        }
    }
}

// remap op (single-threaded on ith 0): make every expert the router picked resident - reserving
// slots and demand-loading misses, waiting until they commit - then rewrite each id to its cache
// slot. Only the ids change, so the same experts are computed in the same order.
void llama_moe_stream_remap(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    // timed from before the lock: the GPU is idle while this CPU op runs
    const int64_t t_op0 = ggml_time_us();

    auto * sl  = (llama_moe_stream_layer *) userdata;
    auto * mgr = sl->mgr;

    GGML_ASSERT(a->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_are_same_shape(a, dst));

    const int64_t n = ggml_nelements(a);

    const int32_t * ids = (const int32_t *) a->data;
          int32_t * out = (int32_t *) dst->data;

    std::unique_lock<std::mutex> lk(mgr->mtx);

    mgr->maybe_dump_stats_locked();

    if (mgr->load_failed) {
        GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
    }

    mgr->stats.n_calls++;
    mgr->start_workers_locked();

    // distinct experts touched by this ubatch, in first-use order
    sl->touched.assign(sl->n_expert, 0);
    sl->uniq.clear();
    for (int64_t i = 0; i < n; i++) {
        const int32_t e = ids[i];
        GGML_ASSERT(e >= 0 && (uint32_t) e < sl->n_expert);
        if (!sl->touched[e]) {
            sl->touched[e] = 1;
            sl->uniq.push_back(e);
        }
    }

    if (sl->uniq.size() > sl->n_slots) {
        GGML_ABORT("MoE expert streaming: layer %d needs %zu distinct experts but the cache has only %u slots; "
                   "increase --moe-stream-cache or reduce the ubatch size (-ub)",
                sl->il, sl->uniq.size(), sl->n_slots);
    }

    // route hotness for eviction; halved periodically so a formerly hot expert ages out
    for (const int32_t e : sl->uniq) {
        sat_inc(sl->route_hotness[e]);
    }
    if (mgr->hot_decay_interval > 0 && mgr->stats.n_calls % mgr->hot_decay_interval == 0) {
        for (auto & sl2 : mgr->layers) {
            if (sl2) {
                for (auto & h : sl2->route_hotness) {
                    h >>= 1;
                }
            }
        }
    }

    // classify the touched experts; reserve and enqueue demand loads in deterministic order
    std::fill(sl->keep.begin(), sl->keep.end(), 0);
    sl->demand_slots.clear();

    bool waited = false;
    for (const int32_t e : sl->uniq) {
        const auto it = sl->expert_slot.find(e);
        if (it != sl->expert_slot.end()) {
            const int32_t s = it->second;
            if (sl->slot_state[s] == LLAMA_MOE_STREAM_SLOT_LOADING) {
                mgr->promote_slot_locked(*sl, s);
                waited = true;
                mgr->stats.n_hit_loading++;
            } else {
                mgr->stats.n_hit_ready++;
            }
            mgr->stats.n_hit++;
            sl->keep[s] = 1;
            sl->demand_slots.push_back(s);
        } else {
            int32_t v;
            while ((v = mgr->pick_victim_locked(*sl, sl->keep.data())) < 0) {
                // every allowed slot is loading; wait for a commit and retry
                mgr->cv_done.wait(lk);
                if (mgr->load_failed) {
                    GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
                }
            }
            if (!sl->seen[e]) {
                mgr->stats.n_miss_cold++;
            }
            const int32_t old = sl->slot_expert[v];
            if (old >= 0 && sl->touched[old]) {
                mgr->stats.n_demand_self_evict++;
            }
            mgr->reserve_slot_locked(*sl, e, v);
            // one work item PER SLAB: the 2-3 slabs of an expert are independent reads, and issuing
            // them together is what lifts the device queue depth above 1
            sl->slot_pending[v] = (uint8_t) sl->weights.size();
            for (size_t wi = 0; wi < sl->weights.size(); wi++) {
                mgr->q_demand.push_back({ sl, e, v, (int32_t) wi, sl->slot_gen[v], true });
                mgr->cv_work.notify_one();
            }
            mgr->stats.n_miss++;
            waited = true;
            sl->keep[v] = 1;
            sl->demand_slots.push_back(v);
        }
    }

    if (waited) {
        const int64_t t0 = ggml_time_us();
        mgr->cv_done.wait(lk, [&]{
            if (mgr->load_failed) {
                return true;
            }
            for (const int32_t s : sl->demand_slots) {
                if (sl->slot_state[s] != LLAMA_MOE_STREAM_SLOT_RESIDENT) {
                    return false;
                }
            }
            return true;
        });
        if (mgr->load_failed) {
            GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
        }
        mgr->stats.t_stall_us += ggml_time_us() - t0;
    }

    for (int64_t i = 0; i < n; i++) {
        const int32_t s = sl->expert_slot.at(ids[i]);
        sl->slot_last_use[s] = ++sl->use_counter;
        out[i] = s;
    }

    mgr->stats.t_remap_op_us += ggml_time_us() - t_op0;
}

// prefetch next-layer experts without another graph split; wrong guesses cost read bandwidth and
// can evict useful residents, so the candidate budget is bounded
static void llama_moe_stream_prefetch_next(llama_moe_stream_lookahead * la, const float * logits, uint32_t rows) {
    llama_moe_stream_layer & sl = *la->sl_next;
    auto * mgr = sl.mgr;

    llama_moe_lookahead_select(logits, sl.n_expert, rows, la->top_k, la->bias, la->score, la->selected);
    for (const int32_t best : la->selected) {
        if (sl.expert_slot.find(best) != sl.expert_slot.end()) {
            continue; // already resident or in flight - the common case
        }
        if (mgr->q_spec.size() >= mgr->q_spec_max) {
            break; // backlog deeper than the drive clears in time
        }
        const int32_t v = mgr->pick_victim_locked(sl, nullptr);
        if (v < 0) {
            break; // every slot busy; the layer's own remap will demand-load it
        }
        mgr->reserve_slot_locked(sl, best, v);
        sl.slot_pending[v] = (uint8_t) sl.weights.size();
        for (size_t wi = 0; wi < sl.weights.size(); wi++) {
            mgr->q_spec.push_back({ &sl, best, v, (int32_t) wi, sl.slot_gen[v], false });
            mgr->cv_work.notify_one();
        }
        mgr->stats.n_preload_issued++;
    }
}

void llama_moe_stream_remap_la(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b, int ith, int nth, void * userdata) {
    auto * la = (llama_moe_stream_lookahead *) userdata;

    if (ith != 0 || la->sl_next == nullptr || la->top_k == 0) {
        llama_moe_stream_remap(dst, a, ith, nth, la->sl);
        return;
    }

    // b holds every row's prediction; the selection uses the last row
    GGML_ASSERT(b->type == GGML_TYPE_F32 && ggml_is_contiguous(b));
    const int64_t n_tok = b->ne[1] > 0 ? b->ne[1] : 1;
    const float * logits = (const float *) b->data;

    if (!la->bias_read) {
        la->bias_read = true;
        if (la->bias_src) {
            la->bias.resize(ggml_nelements(la->bias_src));
            ggml_backend_tensor_get(la->bias_src, la->bias.data(), 0, ggml_nbytes(la->bias_src));
        }
    }

    llama_moe_stream_remap(dst, a, ith, nth, la->sl);
    std::unique_lock<std::mutex> lk(la->sl_next->mgr->mtx);
    if (!la->sl_next->mgr->load_failed) {
        llama_moe_stream_prefetch_next(la, logits, (uint32_t) n_tok);
    }
}

// wave 0 of a ubatch: record the distinct touched experts (sl.uniq, first-use order) and split
// them into the waves the graph built
void llama_moe_stream::plan_waves_locked(llama_moe_stream_layer & sl, const int32_t * ids, int64_t n) {
    stats.n_calls++;
    start_workers_locked();
    maybe_dump_stats_locked();

    sl.touched.assign(sl.n_expert, 0);
    sl.uniq.clear();
    for (int64_t i = 0; i < n; i++) {
        const int32_t e = ids[i];
        GGML_ASSERT(e >= 0 && (uint32_t) e < sl.n_expert);
        if (!sl.touched[e]) {
            sl.touched[e] = 1;
            sl.uniq.push_back(e);
        }
    }

    // no hotness update here: a prefill sweep touches nearly every expert and would flatten the
    // decode hotness that keeps the right experts resident afterwards

    GGML_ASSERT(sl.plan_capacity > 0);
    sl.expert_wave.assign(sl.n_expert, 0xff);
    for (size_t i = 0; i < sl.uniq.size(); i++) {
        GGML_ASSERT(i/sl.plan_capacity < 0xff);
        sl.expert_wave[sl.uniq[i]] = (uint8_t) (i/sl.plan_capacity);
    }
    sl.plan_n_waves   = (uint32_t) ((sl.uniq.size() + sl.plan_capacity - 1)/sl.plan_capacity);
    sl.plan_next_wave = 0;

    // the partition path needs exactly the wave count the graph built: fewer, fatter waves would
    // overflow the pair chunk the graph sized for that count
    const size_t n_uniq = sl.uniq.size();
    if (sl.plan_waves_want > 1 && n_uniq >= sl.plan_waves_want) {
        sl.plan_n_waves = sl.plan_waves_want;
    }
    sl.wave_first.assign(sl.plan_n_waves, 0);
    sl.wave_count.assign(sl.plan_n_waves, 0);
    {
        // spread the experts evenly over exactly plan_n_waves slices, never above plan_capacity
        const size_t base = sl.plan_n_waves > 0 ? n_uniq/sl.plan_n_waves : 0;
        const size_t rem  = sl.plan_n_waves > 0 ? n_uniq%sl.plan_n_waves : 0;
        size_t at = 0;
        for (uint32_t w = 0; w < sl.plan_n_waves; w++) {
            const size_t cnt = std::min<size_t>(base + (w < rem ? 1 : 0), sl.plan_capacity);
            sl.wave_first[w] = (uint32_t) at;
            sl.wave_count[w] = (uint32_t) cnt;
            at += cnt;
        }
        GGML_ASSERT(at == n_uniq); // every touched expert belongs to exactly one wave
        for (uint32_t w = 0; w < sl.plan_n_waves; w++) {
            for (uint32_t i = 0; i < sl.wave_count[w]; i++) {
                sl.expert_wave[sl.uniq[sl.wave_first[w] + i]] = (uint8_t) w;
            }
        }
    }

    // keyed off the chunk the graph set: a ubatch too small to partition runs the masked path
    if (sl.plan_pair_chunk > 0) {
        plan_pairs_locked(sl, ids, n);
    }
}

// pair partitioning: give every (token, expert) pair to exactly one wave, balancing the pair count
// across waves because the graph fixed the per-wave chunk before the router ran
void llama_moe_stream::plan_pairs_locked(llama_moe_stream_layer & sl, const int32_t * ids, int64_t n) {
    const uint32_t n_waves = sl.plan_n_waves;
    if (n_waves <= 1) {
        sl.plan_pair.clear();
        return;
    }

    // pairs per expert - the weight each expert brings to its wave
    sl.pair_count.assign(sl.n_expert, 0);
    for (int64_t i = 0; i < n; i++) {
        sl.pair_count[ids[i]]++;
    }

    const std::vector<uint32_t> room(sl.wave_count);

    // heaviest expert first
    std::vector<int32_t> order(sl.uniq);
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        return sl.pair_count[a] > sl.pair_count[b];
    });

    // snake order over the waves (0, 1, .., k-1, k-1, .., 0, ..): pairs each heavy expert with a light
    // one and fills every wave to exactly its expert count. Greedy-lightest fills the light waves'
    // expert slots first and then has to force the rest regardless of load.
    std::vector<std::vector<int32_t>> group(n_waves);
    std::vector<int64_t>              load (n_waves, 0);
    {
        std::vector<uint32_t> left(room);
        std::vector<uint32_t> seq;
        seq.reserve(order.size());

        uint32_t w = 0;
        int      dir = 1;
        while (seq.size() < order.size()) {
            if (left[w] > 0) {
                seq.push_back(w);
                left[w]--;
            }
            if (dir > 0) {
                if (w + 1 < n_waves) { w++; } else { dir = -1; }
            } else {
                if (w > 0) { w--; } else { dir = 1; }
            }
        }
        for (size_t i = 0; i < order.size(); i++) {
            group[seq[i]].push_back(order[i]);
            load [seq[i]] += sl.pair_count[order[i]];
        }
    }

    // repair pass: swap a heavy expert of the worst wave for a lighter one of the best wave
    // (keeps both expert counts); only while a wave is over the chunk
    for (int iter = 0; iter < 64; iter++) {
        uint32_t hi = 0, lo = 0;
        for (uint32_t w = 1; w < n_waves; w++) {
            if (load[w] > load[hi]) { hi = w; }
            if (load[w] < load[lo]) { lo = w; }
        }
        if (load[hi] <= (int64_t) sl.plan_pair_chunk || hi == lo) {
            break;
        }

        const int64_t gap = load[hi] - load[lo];
        int64_t best_d = 0;
        size_t  bi = 0, bj = 0;
        for (size_t i = 0; i < group[hi].size(); i++) {
            for (size_t j = 0; j < group[lo].size(); j++) {
                const int64_t d = sl.pair_count[group[hi][i]] - sl.pair_count[group[lo][j]];
                if (d > best_d && 2*d <= gap + best_d) { best_d = d; bi = i; bj = j; }
            }
        }
        if (best_d <= 0) {
            break;
        }
        std::swap(group[hi][bi], group[lo][bj]);
        load[hi] -= best_d;
        load[lo] += best_d;
    }

    sl.uniq.clear();
    for (uint32_t w = 0; w < n_waves; w++) {
        sl.wave_first[w] = (uint32_t) sl.uniq.size();
        sl.wave_count[w] = (uint32_t) group[w].size();
        for (const int32_t e : group[w]) {
            sl.expert_wave[e] = (uint8_t) w;
            sl.uniq.push_back(e);
        }
    }

    const int64_t mean = (int64_t) n/n_waves;
    for (uint32_t w = 0; w < n_waves; w++) {
        stats.pair_over_max = std::max(stats.pair_over_max, mean > 0 ? (load[w] - mean)*100/mean : 0);
    }

    int64_t worst = 0;
    for (uint32_t w = 0; w < n_waves; w++) {
        worst = std::max(worst, load[w]);
    }
    if (sl.plan_pair_chunk > 0) {
        stats.chunk_util_max = std::max(stats.chunk_util_max, worst*100/(int64_t) sl.plan_pair_chunk);
    }

    // flat pair indices (t*n_ids + k) owned by each wave
    sl.plan_pair.assign(n_waves, {});
    for (uint32_t w = 0; w < n_waves; w++) {
        sl.plan_pair[w].reserve((size_t) load[w]);
    }
    for (int64_t i = 0; i < n; i++) {
        sl.plan_pair[sl.expert_wave[ids[i]]].push_back((int32_t) i);
    }

    // split pass: move surplus pairs off an over-full wave, staging that expert in the receiving
    // wave as well. One expert can hold n_tokens pairs, so balancing whole experts cannot always fit
    // the chunk; staging an expert in a second wave costs a slot there, and its second staging is a
    // cache hit, not I/O.
    const size_t chunk = sl.plan_pair_chunk;
    for (uint32_t w = 0; w < n_waves && chunk > 0; w++) {
        while (sl.plan_pair[w].size() > chunk) {
            const size_t surplus = sl.plan_pair[w].size() - chunk;

            // the expert contributing most to this wave is the one worth moving
            std::unordered_map<int32_t, size_t> cnt;
            for (const int32_t idx : sl.plan_pair[w]) {
                cnt[ids[idx]]++;
            }
            int32_t hot = -1; size_t hot_n = 0;
            for (const auto & kv : cnt) {
                if (kv.second > hot_n) { hot = kv.first; hot_n = kv.second; }
            }
            if (hot < 0) {
                break;
            }

            // a receiver needs pair room, and a free expert slot unless it already stages `hot`;
            // a free move beats a bigger one that costs a slot
            int32_t dst = -1; size_t room_best = 0; bool dst_has = false;
            for (uint32_t v = 0; v < n_waves; v++) {
                if (v == w || sl.plan_pair[v].size() >= chunk) {
                    continue;
                }
                const bool has = std::find(group[v].begin(), group[v].end(), hot) != group[v].end();
                if (!has && group[v].size() >= sl.plan_capacity) {
                    continue;
                }
                const size_t r = chunk - sl.plan_pair[v].size();
                if (has != dst_has ? has : r > room_best) { room_best = r; dst = (int32_t) v; dst_has = has; }
            }
            if (dst < 0) {
                break; // no receiver; the check below reports it
            }

            const size_t move = std::min({surplus, room_best, hot_n});
            std::vector<int32_t> kept;
            kept.reserve(sl.plan_pair[w].size() - move);
            size_t moved = 0;
            for (const int32_t idx : sl.plan_pair[w]) {
                if (moved < move && ids[idx] == hot) { sl.plan_pair[dst].push_back(idx); moved++; }
                else                                  { kept.push_back(idx); }
            }
            sl.plan_pair[w].swap(kept);
            if (!dst_has) {
                group[dst].push_back(hot); // stage it in the receiver too
            }
            stats.n_pair_splits++;
            if (moved == 0) {
                break;
            }
        }
    }

    // uniq must reflect the split staging
    sl.uniq.clear();
    for (uint32_t w = 0; w < n_waves; w++) {
        sl.wave_first[w] = (uint32_t) sl.uniq.size();
        sl.wave_count[w] = (uint32_t) group[w].size();
        for (const int32_t e : group[w]) {
            sl.uniq.push_back(e);
        }
    }

    for (uint32_t w = 0; w < n_waves; w++) {
        if (sl.plan_pair[w].size() > sl.plan_pair_chunk) {
            size_t free_pairs = 0, free_slots = 0;
            for (uint32_t v = 0; v < n_waves; v++) {
                free_pairs += sl.plan_pair[v].size() < chunk                   ? 1 : 0;
                free_slots += group[v].size()        < (size_t) sl.plan_capacity ? 1 : 0;
            }
            GGML_ABORT("MoE expert streaming: wave %u holds %zu pairs but chunk is %u after splitting "
                       "(%u waves, %zu with pair room, %zu with a free expert slot, capacity %u, %zu experts staged)",
                    w, sl.plan_pair[w].size(), sl.plan_pair_chunk,
                    n_waves, free_pairs, free_slots, sl.plan_capacity, sl.uniq.size());
        }
    }
}

// make wave w's experts resident (waiting for their loads) and best-effort preload the next wave so
// its loads overlap this wave's compute. Leaves sl.demand_slots = this wave's slots and sl.plan_pool
// = the resident parking pool (>= n_ids slots) the masked pairs draw from
void llama_moe_stream::stage_wave_locked(std::unique_lock<std::mutex> & lk, llama_moe_stream_layer & sl, int32_t w, uint32_t n_ids) {
    const size_t first = (size_t) w < sl.wave_first.size() ? sl.wave_first[w] : sl.uniq.size();
    const size_t count = (size_t) w < sl.wave_count.size() ? sl.wave_count[w] : 0;

    std::fill(sl.keep.begin(), sl.keep.end(), 0);
    sl.demand_slots.clear();

    // a small final wave has fewer than n_ids own slots; borrow the rest from the previous wave's
    // pool so every token row has n_ids distinct resident parking slots
    std::vector<int32_t> borrowed;
    if (count < n_ids) {
        GGML_ASSERT(sl.plan_pool.size() >= n_ids - count);
        for (size_t i = 0; i < n_ids - count; i++) {
            borrowed.push_back(sl.plan_pool[i]);
            sl.keep[sl.plan_pool[i]] = 1;
        }
    }

    // protect the next wave's resident experts from this wave's victims, bounded by the slack that
    // is really there: an unbounded keep-set can cover every slot and hang the demand loop below
    const size_t nw     = (size_t) w + 1;
    const size_t nfirst = nw < sl.wave_first.size() ? sl.wave_first[nw] : sl.uniq.size();
    const size_t ncount = nw < sl.wave_count.size() ? sl.wave_count[nw] : 0;

    const size_t reserved = (size_t) sl.plan_capacity + n_ids + 1; // this wave, parking, one victim
    const size_t max_keep = sl.n_slots > reserved ? sl.n_slots - reserved : 0;

    size_t n_kept = 0;
    for (size_t i = nfirst; i < nfirst + ncount && n_kept < max_keep; i++) {
        const auto it = sl.expert_slot.find(sl.uniq[i]);
        if (it != sl.expert_slot.end()) {
            sl.keep[it->second] = 1;
            n_kept++;
        }
    }

    bool waited = false;
    if (count > 0) {
        stats.n_waves_run++;
        for (size_t i = first; i < first + count; i++) {
            const int32_t e  = sl.uniq[i];
            const auto    it = sl.expert_slot.find(e);
            if (it != sl.expert_slot.end()) {
                const int32_t s = it->second;
                if (sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_LOADING) {
                    promote_slot_locked(sl, s);
                    waited = true;
                } else {
                    stats.n_preload_ready++;
                }
                stats.n_hit++;
                sl.keep[s] = 1;
                sl.demand_slots.push_back(s);
            } else {
                int32_t v;
                while ((v = pick_victim_locked(sl, sl.keep.data())) < 0) {
                    cv_done.wait(lk);
                    if (load_failed) {
                        GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
                    }
                }
                if (!sl.seen[e]) {
                    stats.n_miss_cold++;
                }
                reserve_slot_locked(sl, e, v);
                sl.slot_pending[v] = (uint8_t) sl.weights.size();
                for (size_t wi = 0; wi < sl.weights.size(); wi++) {
                    q_demand.push_back({ &sl, e, v, (int32_t) wi, sl.slot_gen[v], true });
                    cv_work.notify_one();
                }
                stats.n_miss++;
                waited = true;
                sl.keep[v] = 1;
                sl.demand_slots.push_back(v);
            }
        }
    }

    // best-effort preload of the next wave; never waits, what cannot be reserved now becomes the
    // next wave's demand load
    static const bool no_preload = std::getenv("LLAMA_MOE_STREAM_NO_PRELOAD") != nullptr;
    if (!no_preload) {
        for (size_t i = nfirst; i < nfirst + ncount; i++) {
            const int32_t e = sl.uniq[i];
            if (sl.expert_slot.find(e) != sl.expert_slot.end()) {
                continue;
            }
            const int32_t v = pick_victim_locked(sl, sl.keep.data());
            if (v < 0) {
                continue;
            }
            if (!sl.seen[e]) {
                stats.n_miss_cold++;
            }
            reserve_slot_locked(sl, e, v);
            sl.keep[v] = 1;
            sl.slot_pending[v] = (uint8_t) sl.weights.size();
            for (size_t wi = 0; wi < sl.weights.size(); wi++) {
                q_demand.push_back({ &sl, e, v, (int32_t) wi, sl.slot_gen[v], true });
                cv_work.notify_one();
            }
            stats.n_preload_issued++;
        }
    }

    if (waited) {
        const int64_t t0 = ggml_time_us();
        cv_done.wait(lk, [&]{
            if (load_failed) {
                return true;
            }
            for (const int32_t s : sl.demand_slots) {
                if (sl.slot_state[s] != LLAMA_MOE_STREAM_SLOT_RESIDENT) {
                    return false;
                }
            }
            return true;
        });
        if (load_failed) {
            GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
        }
        stats.t_stall_wave_us += ggml_time_us() - t0;
    }

    // parking pool: this wave's resident slots plus the borrowed ones, all keep-protected; the next
    // reservation of this layer is ordered after this wave's GEMMs by the graph
    sl.plan_pool = sl.demand_slots;
    sl.plan_pool.insert(sl.plan_pool.end(), borrowed.begin(), borrowed.end());
    GGML_ASSERT(sl.plan_pool.size() >= n_ids);
}

// out[i] = the slot the GEMM indexes for each (token, expert) pair of wave w, one token row at a
// time: pairs of this wave get their real slot, the rest park on distinct resident pool slots
// (a slot must not repeat within a row for the Metal kernel)
void llama_moe_stream::emit_wave_slots(llama_moe_stream_layer & sl, const int32_t * ids, int32_t * out,
        int32_t w, uint32_t n_ids, int64_t n_tok) {
    for (int64_t t = 0; t < n_tok; t++) {
        sl.pool_used.clear();

        for (uint32_t kk = 0; kk < n_ids; kk++) {
            const int64_t i = t*n_ids + kk;
            const int32_t e = ids[i];
            GGML_ASSERT(sl.expert_wave[e] != 0xff);
            if (sl.expert_wave[e] == (uint8_t) w) {
                const int32_t s = sl.expert_slot.at(e);
                GGML_ASSERT(sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_RESIDENT);
                sl.slot_last_use[s] = ++sl.use_counter;
                out[i] = s;
                sl.pool_used.push_back(s);
            }
        }

        size_t pi = 0;
        for (uint32_t kk = 0; kk < n_ids; kk++) {
            const int64_t i = t*n_ids + kk;
            if (sl.expert_wave[ids[i]] == (uint8_t) w) {
                continue;
            }
            while (std::find(sl.pool_used.begin(), sl.pool_used.end(), sl.plan_pool[pi]) != sl.pool_used.end()) {
                pi++;
                GGML_ASSERT(pi < sl.plan_pool.size());
            }
            GGML_ASSERT(sl.slot_state[sl.plan_pool[pi]] == LLAMA_MOE_STREAM_SLOT_RESIDENT);
            out[i] = sl.plan_pool[pi];
            sl.pool_used.push_back(sl.plan_pool[pi]);
            pi++;
        }
    }
}

// partition padding may use slot -1 when every expert cache is a Metal buffer: the Metal mul_mm_id
// map drops an id that matches no slot, so the padding costs no GEMM tiles and its rows go to the
// scratch row. Other backends index weights by the id directly and keep repeated pairs.
bool llama_moe_stream::pad_skip_ok() {
    if (pad_skip < 0) {
        const char * e = getenv("LLAMA_MOE_STREAM_PAD_SKIP");
        bool ok = !(e && atoi(e) == 0);
        bool any = false;
        for (const auto & group : cache_bufs) {
            if (!group.buf) {
                continue;
            }
            any = true;
#ifdef GGML_USE_METAL
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(group.buf.get()));
            ok = ok && dev != nullptr && ggml_backend_dev_backend_reg(dev) == ggml_backend_metal_reg();
#else
            ok = false;
#endif
        }
        pad_skip = ok && any ? 1 : 0;
    }
    return pad_skip > 0;
}

// partition path: write the four index rows of wave w's dense pair list, padded to the static chunk
// by repeating the wave's last pair (same value to the same row), or with slot -1 to the scratch
// row on Metal. A wave that owns no pair computes a throwaway row on a parked slot and scatters it
// to the scratch row past the real pairs, which nothing reads.
void llama_moe_stream::emit_wave_pairs(llama_moe_stream_layer & sl, const int32_t * ids, int32_t * out,
        int32_t w, uint32_t n_ids, int64_t n_pairs, int64_t chunk) {
    const std::vector<int32_t> * pairs = (size_t) w < sl.plan_pair.size() ? &sl.plan_pair[w] : nullptr;
    if (pairs != nullptr && pairs->empty()) {
        pairs = nullptr;
    }
    GGML_ASSERT(pairs == nullptr || (int64_t) pairs->size() <= chunk);

    int32_t * r_tok  = out + LLAMA_MOE_PAIR_TOK *chunk;
    int32_t * r_pair = out + LLAMA_MOE_PAIR_PAIR*chunk;
    int32_t * r_slot = out + LLAMA_MOE_PAIR_SLOT*chunk;
    int32_t * r_exp  = out + LLAMA_MOE_PAIR_EXP *chunk;

    // below 32 rows Metal takes the mul_mv_id path, which reads the slot id without a check
    const bool skip = chunk >= 32 && pad_skip_ok();

    if (pairs == nullptr) {
        GGML_ASSERT(!sl.plan_pool.empty());
        const int32_t s = sl.plan_pool[0];
        GGML_ASSERT(sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_RESIDENT);
        for (int64_t p = 0; p < chunk; p++) {
            r_tok [p] = 0;
            r_pair[p] = (int32_t) n_pairs; // scratch row
            r_slot[p] = skip ? -1 : s;
            r_exp [p] = sl.slot_expert[s];
        }
        return;
    }

    for (int64_t p = 0; p < chunk; p++) {
        const bool pad = p >= (int64_t) pairs->size();
        const int32_t i = (*pairs)[pad ? pairs->size() - 1 : (size_t) p];
        const int32_t e = ids[i];
        const int32_t s = sl.expert_slot.at(e);
        GGML_ASSERT(sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_RESIDENT);
        sl.slot_last_use[s] = ++sl.use_counter;

        r_tok [p] = i/(int32_t) n_ids;
        r_pair[p] = pad && skip ? (int32_t) n_pairs : i;
        r_slot[p] = pad && skip ? -1 : s;
        r_exp [p] = e;
    }
}

// shared preamble of the wave ops: plan the ubatch on wave 0, enforce the wave order, and make wave
// w resident while preloading the next. Returns with the mutex held for the emit that follows.
static const size_t   LLAMA_MOE_STREAM_PLAN_OFFS  = 32;
static const uint32_t LLAMA_MOE_STREAM_PLAN_MAGIC = 0x4e414c50; // "PLAN"

void llama_moe_stream_set_plan(ggml_tensor * op, uint32_t pair_chunk, uint32_t waves_want, uint32_t capacity) {
    static_assert(LLAMA_MOE_STREAM_PLAN_OFFS + 16 <= GGML_MAX_OP_PARAMS, "op_params too small for the wave plan");
    const uint32_t v[4] = { pair_chunk, waves_want, capacity, LLAMA_MOE_STREAM_PLAN_MAGIC };
    memcpy((char *) op->op_params + LLAMA_MOE_STREAM_PLAN_OFFS, v, sizeof(v));
}

static std::unique_lock<std::mutex> stage_wave_for_op(llama_moe_stream_layer & sl, int32_t w,
        const int32_t * ids, int64_t n, uint32_t n_ids, const ggml_tensor * dst) {
    auto * mgr = sl.mgr;

    std::unique_lock<std::mutex> lk(mgr->mtx);

    if (mgr->load_failed) {
        GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
    }

    mgr->stats.n_wave_calls++;

    // this graph's plan, not whatever graph was built last (see llama_moe_stream_set_plan)
    uint32_t v[4];
    memcpy(v, (const char *) dst->op_params + LLAMA_MOE_STREAM_PLAN_OFFS, sizeof(v));
    GGML_ASSERT(v[3] == LLAMA_MOE_STREAM_PLAN_MAGIC && "wave op built without its plan");
    sl.plan_pair_chunk = v[0];
    sl.plan_waves_want = v[1];
    sl.plan_capacity   = v[2];

    if (w == 0) {
        mgr->plan_waves_locked(sl, ids, n);
    }
    GGML_ASSERT(sl.plan_next_wave == w); // waves run in order (enforced by the graph ordering token)

    mgr->stage_wave_locked(lk, sl, w, n_ids);
    sl.plan_next_wave = w + 1;

    return lk;
}

void llama_moe_stream_wave_ids(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    const int64_t t_op0 = ggml_time_us();

    auto * ud  = (llama_moe_stream_wave *) userdata;
    auto * sl  = ud->sl;
    auto * mgr = sl->mgr;

    const int32_t w = ud->wave;

    const ggml_tensor * a = dst->src[0]; // contiguous selected ids
    GGML_ASSERT(a->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(a));
    GGML_ASSERT(dst->data != a->data); // the emit must not clobber the ids other waves read

    const int64_t   n   = ggml_nelements(a);
    const int32_t * ids = (const int32_t *) a->data;
          int32_t * out = (int32_t *) dst->data;

    std::unique_lock<std::mutex> lk = stage_wave_for_op(*sl, w, ids, n, (uint32_t) a->ne[0], dst);

    mgr->emit_wave_slots(*sl, ids, out, w, (uint32_t) a->ne[0], a->ne[1]);

    mgr->stats.t_wave_op_us += ggml_time_us() - t_op0;
}

void llama_moe_stream_wave_pairs(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    const int64_t t_op0 = ggml_time_us();

    auto * ud  = (llama_moe_stream_wave *) userdata;
    auto * sl  = ud->sl;
    auto * mgr = sl->mgr;

    const int32_t w = ud->wave;

    const ggml_tensor * a = dst->src[0]; // contiguous selected ids
    GGML_ASSERT(a->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(dst->ne[1] == LLAMA_MOE_PAIR_ROWS);

    const int64_t   n   = ggml_nelements(a);
    const int32_t * ids = (const int32_t *) a->data;
          int32_t * out = (int32_t *) dst->data;
    if (ids == nullptr || out == nullptr) {
        GGML_ABORT("wave_pairs: null data: a=%s %p buf=%p, dst=%s %p buf=%p, view_src=%p",
                a->name, a->data, (void *) a->buffer, dst->name, dst->data, (void *) dst->buffer, (void *) a->view_src);
    }

    std::unique_lock<std::mutex> lk = stage_wave_for_op(*sl, w, ids, n, (uint32_t) a->ne[0], dst);

    mgr->emit_wave_pairs(*sl, ids, out, w, (uint32_t) a->ne[0], n, dst->ne[0]);

    mgr->stats.t_wave_op_us += ggml_time_us() - t_op0;
}

void llama_moe_stream_wave_mask(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    auto * ud  = (llama_moe_stream_wave *) userdata;
    auto * sl  = ud->sl;
    auto * mgr = sl->mgr;

    const int32_t w = ud->wave;

    const ggml_tensor * a = dst->src[0]; // contiguous selected ids
    GGML_ASSERT(a->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(a));

    const int64_t   n   = ggml_nelements(a);
    const int32_t * ids = (const int32_t *) a->data;
          float   * out = (float *) dst->data;

    std::lock_guard<std::mutex> lock(mgr->mtx);

    GGML_ASSERT(sl->plan_next_wave > w); // this wave's ids op has already run

    for (int64_t i = 0; i < n; i++) {
        out[i] = sl->expert_wave[ids[i]] == (uint8_t) w ? 1.0f : 0.0f;
    }
}
