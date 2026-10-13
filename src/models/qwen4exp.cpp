#include "models.h"
#include "llama-impl.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-memory-recurrent.h"
#include "llama-moe-stream.h"

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <map>

#if defined(__APPLE__) || defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#define QWEN4EXP_PLE_PREFETCH 1
#endif

// bad metadata must be catchable: GGML_ASSERT aborts the whole process
static void qwen4exp_require_nonzero(const llama_model_loader & ml, llm_kv kid, uint32_t value) {
    if (value == 0) {
        throw std::runtime_error(format("%s must be greater than zero, got %u", ml.llm_kv(kid).c_str(), value));
    }
}

// get_arr() copies a short array as-is, leaving a zero tail the n-gram hash silently drops
static void qwen4exp_require_arr_len(llama_model_loader & ml, llm_kv kid, uint32_t n_min) {
    uint32_t n_arr = 0;
    ml.get_arr_n(kid, n_arr, true);
    if (n_arr < n_min) {
        throw std::runtime_error(format("%s has %u entries, but at least %u are required",
                                        ml.llm_kv(kid).c_str(), n_arr, n_min));
    }
}

static const llama_model & qwen4exp_shared_model(const llama_cparams & cparams, const llama_model & model, const char * name) {
    if (cparams.ctx_other == nullptr) {
        throw std::runtime_error(format("QWEN4EXP MTP: this draft head has no '%s' of its own; "
                                        "load it as a draft of its target model (-md), not on its own", name));
    }
    const llama_model & other = *llama_get_model(cparams.ctx_other);
    if (other.hparams.n_embd != model.hparams.n_embd || other.vocab.n_tokens() != model.vocab.n_tokens()) {
        throw std::runtime_error(format("QWEN4EXP MTP: draft and target disagree on the shape of '%s'", name));
    }
    return other;
}

void llama_model_qwen4exp::load_arch_hparams(llama_model_loader & ml) {
    // the trunk must keep at least one block: n_layer() == n_layer_all - n_layer_nextn
    if (hparams.n_layer_nextn >= hparams.n_layer_all) {
        throw std::runtime_error(format("%s must be less than %s, got %u",
                                        ml.llm_kv(LLM_KV_NEXTN_PREDICT_LAYERS).c_str(),
                                        ml.llm_kv(LLM_KV_BLOCK_COUNT).c_str(), hparams.n_layer_nextn));
    }

    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // HC; low_rank is qwen4exp-specific, DeepSeek-V4 leaves it absent (full rank)
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,    hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    // a count of 1 has nothing to mix: transformers configuration_qwen4_exp.py:196, vLLM
    // config.py:49 and SGLang configs/qwen4_exp.py:38 all raise on hc_count <= 1
    if (hparams.dsv4_hc_mult <= 1) {
        throw std::runtime_error(format("%s must be greater than one, got %u",
                                        ml.llm_kv(LLM_KV_HYPER_CONNECTION_COUNT).c_str(), hparams.dsv4_hc_mult));
    }
    qwen4exp_require_nonzero(ml, LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    hparams.n_embd_out_impl = hparams.dsv4_hc_mult * hparams.n_embd;

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    ml.get_key_or_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.dsv4_compress_ratios, hparams.n_layer_all, false);

    // Some community GGUFs (e.g. the i1 Uncensored quants) were converted before QSA existed and
    // store compress ratio 0 everywhere, which silently runs dense attention over the whole
    // context. LLAMA_QWEN4EXP_COMPRESS_RATIO=r restores the reference ratio (indexer_compress_ratio
    // in the HF config, 4 for Qwen3.8-Flash-Next) on the full-attention layers of such files.
    if (const char * s = getenv("LLAMA_QWEN4EXP_COMPRESS_RATIO")) {
        const uint32_t r = (uint32_t) atoi(s);
        bool all_zero = true;
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            all_zero = all_zero && hparams.dsv4_compress_ratios[i] == 0;
        }
        if (r > 0 && all_zero) {
            uint32_t full_attn_interval = 4;
            ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
            for (uint32_t i = 0; i < hparams.n_layer(); ++i) {
                if ((i + 1) % full_attn_interval == 0) {
                    hparams.dsv4_compress_ratios[i] = r;
                }
            }
            LLAMA_LOG_WARN("%s: compress ratios were all 0; using %u on the full-attention layers (LLAMA_QWEN4EXP_COMPRESS_RATIO)\n", __func__, r);
        }
    }

    // QSA pools the indexer keys of blocks of compress_ratio cells, one block size for the whole model
    hparams.indexer_kpool = 0;
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        const uint32_t r = hparams.dsv4_compress_ratios[il];
        if (r == 0) {
            continue;
        }
        if (hparams.indexer_kpool != 0 && r != hparams.indexer_kpool) {
            throw std::runtime_error(format("QSA layers must share one compress ratio, got %u and %u", hparams.indexer_kpool, r));
        }
        hparams.indexer_kpool = r;
    }
    if (hparams.indexer_kpool == 1 || (hparams.indexer_kpool > 0 && hparams.indexer_top_k % hparams.indexer_kpool != 0)) {
        throw std::runtime_error(format("QSA needs a compress ratio above 1 that divides the budget, got %u and %u",
                                        hparams.indexer_kpool, hparams.indexer_top_k));
    }
    // the reference groups the visible tokens in cache order and always keeps the tail
    hparams.indexer_kpool_row         = 2; // raw key | pooled key
    hparams.indexer_kpool_by_order    = true;
    hparams.indexer_kpool_select_tail = true;

    // PLE n-gram hash embeddings; if the key group is absent every field stays zero
    hparams.is_ple_impl.reset();
    hparams.ple_n_heads = 0;

    uint32_t n_ple = 0;
    ml.get_arr_n(LLM_KV_PLE_LAYERS, n_ple, false);
    if (n_ple > 0) {
        std::vector<uint32_t> ple_layers;
        ml.get_arr(LLM_KV_PLE_LAYERS, ple_layers);
        if (n_ple != 1) {
            // hparams holds one set of hash constants, so several PLE modules cannot be represented
            throw std::runtime_error(format("%s lists %u layers, but only one PLE layer is supported",
                                            ml.llm_kv(LLM_KV_PLE_LAYERS).c_str(), n_ple));
        }
        for (uint32_t il : ple_layers) {
            if (il >= hparams.n_layer_all) {
                throw std::runtime_error(format("PLE layer %u is out of range", il));
            }
            hparams.is_ple_impl.set(il);
        }

        ml.get_key(LLM_KV_PLE_NGRAM_SIZE,      hparams.ple_ngram_size);
        ml.get_key(LLM_KV_PLE_HEADS_PER_NGRAM, hparams.ple_heads_per_ngram);
        ml.get_key(LLM_KV_PLE_CONV_KERNEL,     hparams.ple_conv_kernel);
        ml.get_key(LLM_KV_PLE_EOS_TOKEN_ID,    hparams.ple_eos_token_id);
        // optional: files written before this key fall back to the EOS token
        ml.get_key(LLM_KV_PLE_IMAGE_TOKEN_ID,  hparams.ple_image_token_id, false);
        ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, hparams.n_embd_per_layer);
        qwen4exp_require_nonzero(ml, LLM_KV_PLE_CONV_KERNEL,             hparams.ple_conv_kernel);
        qwen4exp_require_nonzero(ml, LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  hparams.n_embd_per_layer);

        hparams.ple_n_heads  = (hparams.ple_ngram_size - 1) * hparams.ple_heads_per_ngram;
        hparams.ple_head_dim = hparams.n_embd_per_layer;
        if (hparams.ple_ngram_size < 2 || hparams.ple_ngram_size > LLAMA_MAX_PLE_NGRAM) {
            throw std::runtime_error(format("PLE n-gram size %u is out of range", hparams.ple_ngram_size));
        }
        if (hparams.ple_n_heads == 0 || hparams.ple_n_heads > LLAMA_MAX_PLE_HEADS) {
            throw std::runtime_error(format("PLE head count %u is out of range", hparams.ple_n_heads));
        }

        qwen4exp_require_arr_len(ml, LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_ngram_size);
        qwen4exp_require_arr_len(ml, LLM_KV_PLE_HEAD_OFFSETS,      hparams.ple_n_heads);
        qwen4exp_require_arr_len(ml, LLM_KV_PLE_HEAD_VOCAB_SIZES,  hparams.ple_n_heads);

        ml.get_arr(LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_layer_multipliers);

        // the file stores the head ranges as uint64, so read at that width and narrow to the int32 the gather uses
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_offsets     = {};
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_vocab_sizes = {};
        ml.get_arr(LLM_KV_PLE_HEAD_OFFSETS,     head_offsets);
        ml.get_arr(LLM_KV_PLE_HEAD_VOCAB_SIZES, head_vocab_sizes);
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            if (head_vocab_sizes[h] == 0 ||
                head_offsets[h]     > INT32_MAX ||
                head_vocab_sizes[h] > INT32_MAX ||
                head_offsets[h] + head_vocab_sizes[h] > INT32_MAX) {
                throw std::runtime_error(format("PLE head %u range does not fit the int32 row index", h));
            }
            hparams.ple_head_offsets[h]     = (uint32_t) head_offsets[h];
            hparams.ple_head_vocab_sizes[h] = (uint32_t) head_vocab_sizes[h];
        }
    }

    // linear attention everywhere except every full_attention_interval-th layer
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        qwen4exp_require_nonzero(ml, LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    // the PLE conv history is a row of the recurrent cache, which linear layers alone have
    for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
        if (hparams.is_ple(i) && !hparams.is_recr(i)) {
            throw std::runtime_error(format("PLE layer %u is not a linear attention layer", i));
        }
    }

    switch (hparams.n_layer()) {
        case 48: type = LLM_TYPE_A3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen4exp::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t hc_lr  = hparams.hc_low_rank;

    const bool mtp_only    = (hparams.n_layer_nextn > 0) && (ml.get_weight("blk.0.hc_attn_norm.weight") == nullptr);
    const int  trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, trunk_flags);

    hc_head_norm = create_tensor(tn(LLM_TENSOR_HC_HEAD_NORM, "weight"), { n_embd, hc }, trunk_flags | TENSOR_ALLOW_RESHAPE);
    hc_head_down = create_tensor(tn(LLM_TENSOR_HC_HEAD_DOWN, "weight"), { hc_dim, hc_lr }, trunk_flags);
    hc_head_up   = create_tensor(tn(LLM_TENSOR_HC_HEAD_UP,   "weight"), { hc_lr, hc_dim }, trunk_flags);

    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    // tie_word_embeddings is false here: never tie to a token_embd a borrowing draft lacks.
    if (output == NULL && tok_embd != NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    // flat [ple_head_dim, n_rows] gather target
    if (hparams.ple_n_heads > 0) {
        // the head ranges are what the gather indexes, so they set the minimum row count
        int64_t ple_rows = 0;
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            ple_rows = std::max(ple_rows, (int64_t) hparams.ple_head_offsets[h] + hparams.ple_head_vocab_sizes[h]);
        }

        // the converter pads the table; a model synthesised from metadata has no tensor to ask
        const std::string ple_name = tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str();
        if (const auto * ple_w = ml.get_weight(ple_name.c_str())) {
            if (ple_w->tensor->ne[1] < ple_rows) {
                throw std::runtime_error(format("%s has %" PRId64 " rows, too few for the PLE head ranges (%" PRId64 ")",
                                                ple_name.c_str(), ple_w->tensor->ne[1], ple_rows));
            }
            ple_rows = ple_w->tensor->ne[1];
        }

        per_layer_tok_embd = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"),
                                           { hparams.ple_head_dim, ple_rows }, TENSOR_READ_LAZY);
    }

    const int mtp_flags = !ml.load_mtp ? TENSOR_SKIP : 0;

    for (int il = 0; il < (int) hparams.n_layer_all; ++il) {
        auto & layer = layers[il];

        const int flags = il < n_layer ? trunk_flags : mtp_flags;

        const int64_t n_ff_exp   = hparams.n_ff_exp() ? hparams.n_ff_exp() : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        // two HC modules per layer: before the token mixer, before the MoE
        layer.hc_attn_norm   = create_tensor(tn(LLM_TENSOR_HC_ATTN_NORM,   "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
        layer.hc_attn_up     = create_tensor(tn(LLM_TENSOR_HC_ATTN_UP,     "weight", il), { hc_lr, hc_dim }, flags);
        layer.hc_ffn_norm    = create_tensor(tn(LLM_TENSOR_HC_FFN_NORM,    "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
        layer.hc_ffn_up      = create_tensor(tn(LLM_TENSOR_HC_FFN_UP,      "weight", il), { hc_lr, hc_dim }, flags);

        // down and inject read the same input: a merged matrix saves one small matmul per HC module
        layer.hc_attn_down_inject = create_tensor(tn(LLM_TENSOR_HC_ATTN_DOWN_INJECT, "weight", il), { hc_dim, hc_lr + hc }, flags | TENSOR_NOT_REQUIRED);
        if (!layer.hc_attn_down_inject) {
            layer.hc_attn_down   = create_tensor(tn(LLM_TENSOR_HC_ATTN_DOWN,   "weight", il), { hc_dim, hc_lr }, flags);
            layer.hc_attn_inject = create_tensor(tn(LLM_TENSOR_HC_ATTN_INJECT, "weight", il), { hc_dim, hc }, flags);
        }
        layer.hc_ffn_down_inject = create_tensor(tn(LLM_TENSOR_HC_FFN_DOWN_INJECT, "weight", il), { hc_dim, hc_lr + hc }, flags | TENSOR_NOT_REQUIRED);
        if (!layer.hc_ffn_down_inject) {
            layer.hc_ffn_down    = create_tensor(tn(LLM_TENSOR_HC_FFN_DOWN,    "weight", il), { hc_dim, hc_lr }, flags);
            layer.hc_ffn_inject  = create_tensor(tn(LLM_TENSOR_HC_FFN_INJECT,  "weight", il), { hc_dim, hc }, flags);
        }

        if (!hparams.is_recr(il)) {
            // full attention: wq holds [q|gate] interleaved per head
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);

            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);

            const int64_t idx_dim = hparams.indexer_head_size;
            layer.index_q_proj = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", il), { n_embd, hparams.indexer_n_head * idx_dim }, flags);
            layer.index_k_proj = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", il), { n_embd, idx_dim }, flags);
            layer.index_q_norm = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", il), { idx_dim }, flags);
            layer.index_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", il), { idx_dim }, flags);
        } else {
            layer.wqkv       = create_tensor(tn(LLM_TENSOR_ATTN_QKV,   "weight", il), { n_embd, key_dim * 2 + value_dim }, flags);
            // z, beta and alpha read the same input: optional merged matrix, rows in this order
            layer.ssm_zba    = create_tensor(tn(LLM_TENSOR_SSM_ZBA,    "weight", il), { n_embd, value_dim + 2 * n_v_heads }, flags | TENSOR_NOT_REQUIRED);
            if (!layer.ssm_zba) {
                layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", il), { n_embd, value_dim }, flags);
                layer.ssm_beta  = create_tensor(tn(LLM_TENSOR_SSM_BETA,  "weight", il), { n_embd, n_v_heads }, flags);
                layer.ssm_alpha = create_tensor(tn(LLM_TENSOR_SSM_ALPHA, "weight", il), { n_embd, n_v_heads }, flags);
            }
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,         il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", il), { head_v_dim }, flags);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", il), { value_dim, n_embd }, flags);
        }

        if (hparams.is_ple(il)) {
            layer.ple_key        = create_tensor(tn(LLM_TENSOR_PLE_KEY,        "weight", il), { n_embd, hc_dim }, flags);
            layer.ple_value      = create_tensor(tn(LLM_TENSOR_PLE_VALUE,      "weight", il), { n_embd, n_embd }, flags);
            layer.ple_norm_key   = create_tensor(tn(LLM_TENSOR_PLE_NORM_KEY,   "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
            layer.ple_norm_query = create_tensor(tn(LLM_TENSOR_PLE_NORM_QUERY, "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
            layer.ple_norm_conv  = create_tensor(tn(LLM_TENSOR_PLE_NORM_CONV,  "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
            layer.ple_conv1d     = create_tensor(tn(LLM_TENSOR_PLE_CONV1D,     "weight", il), { hparams.ple_conv_kernel, hc_dim }, flags);
        }

        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);

        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, flags);
        // optional merged shared-expert gate|up, split by swiglu
        layer.ffn_gate_up_shexp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_SHEXP,  "weight", il), { n_embd, 2 * n_ff_shexp }, flags | TENSOR_NOT_REQUIRED);
        if (!layer.ffn_gate_up_shexp) {
            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, flags);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, flags);
        }
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, flags);

        if (il < n_layer) {
            continue;
        }

        layer.nextn.enorm   = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,   "weight", il), { n_embd }, flags);
        layer.nextn.hnorm   = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,   "weight", il), { hc_dim }, flags);
        layer.nextn.eh_proj = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", il), { 2 * n_embd, n_embd }, flags);

        layer.nextn.hc_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_NORM, "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
        layer.nextn.hc_head_down = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_DOWN, "weight", il), { hc_dim, hc_lr }, flags);
        layer.nextn.hc_head_up   = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_UP,   "weight", il), { hc_lr, hc_dim }, flags);

        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), { n_embd, n_vocab }, flags | TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_vocab }, flags | TENSOR_NOT_REQUIRED);

        // optional reduced-vocabulary draft head (tools/draft_vocab_head.py in NInfer-M1)
        if (const ggml_tensor * sub = ml.get_tensor_meta(tn(LLM_TENSOR_NEXTN_HEAD_SUB, "weight", il).str().c_str())) {
            const int64_t n_sub = sub->ne[1];
            layer.nextn.head_sub = create_tensor(tn(LLM_TENSOR_NEXTN_HEAD_SUB, "weight", il), { n_embd, n_sub }, flags);
            layer.nextn.head_ids = create_tensor(tn(LLM_TENSOR_NEXTN_HEAD_IDS, "weight", il), { n_sub }, flags);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen4exp::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    // without this a self-contained draft loads, then walks the null trunk and segfaults.
    if (hc_head_norm == nullptr) {
        throw std::runtime_error("this model is an MTP draft head without a trunk; "
                                 "load it as a draft of its target model (-md), not on its own");
    }
    return std::make_unique<graph>(*this, params);
}

// NInfer-M1 timing diagnostic: LLAMA_QWEN4EXP_SKIP="gdn,attn,moe,shexp,hcmix" replaces those blocks with
// a pass-through, so the time that disappears is their cost. The output is wrong while it is set.
static bool qwen4exp_skip(const char * what) {
    static const std::string env = [] { const char * s = getenv("LLAMA_QWEN4EXP_SKIP"); return std::string(s ? s : ""); }();
    if (env.empty()) {
        return false;
    }
    return ("," + env + ",").find("," + std::string(what) + ",") != std::string::npos;
}

// Hyper-connections keep hc parallel residual streams [n_embd, hc, T] in place of layer norms.
// Returns the mixed [n_embd, T] stream; `inject` gets the [hc, T] scatter weights.
ggml_tensor * llama_model_qwen4exp::graph::build_hc_mix(
        ggml_tensor *  x,
        ggml_tensor *  w_norm,
        ggml_tensor *  w_down,
        ggml_tensor *  w_up,
        ggml_tensor *  w_inject,
        ggml_tensor ** inject,
        int            il,
        ggml_tensor *  w_down_inject) {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t nt     = x->ne[2];

    if (il >= 0 && qwen4exp_skip("hcmix")) {
        if (inject) {
            *inject = ggml_view_2d(ctx0, x, hc, nt, x->nb[2], 0);
        }
        return ggml_cont(ctx0, ggml_view_2d(ctx0, x, n_embd, nt, x->nb[2], 0));
    }

    // grouped RMSNorm: reduce over one stream, then scale all streams with the [n_embd, hc] gamma
    // the converter folded each gamma to (1 + w)
    ggml_tensor * xn = ggml_mul(ctx0, ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps), w_norm);
    xn = ggml_reshape_2d(ctx0, xn, hc_dim, nt);
    cb(xn, "hc_norm", il);

    // merged weight: one matmul, down rows first, the last hc rows are inject
    ggml_tensor * di = w_down_inject ? build_lora_mm(w_down_inject, xn) : nullptr;
    ggml_tensor * lo = di ? ggml_view_2d(ctx0, di, di->ne[0] - hc, nt, di->nb[1], 0) : build_lora_mm(w_down, xn);
    lo = ggml_silu(ctx0, ggml_scale(ctx0, lo, 1.0f / (float) hc));
    ggml_tensor * gate = build_lora_mm(w_up, lo);
    cb(gate, "hc_gate", il);

    ggml_tensor * mixed = nullptr;
    if (cparams.fused_dsv4_hc_pre && il >= 0) {
        // sigmoid gate and mean over the streams in one op
        mixed = ggml_dsv4_hc_pre_gated(ctx0,
                ggml_reshape_3d(ctx0, xn,   n_embd, hc, nt),
                ggml_reshape_3d(ctx0, gate, n_embd, hc, nt), 1.0f / (float) hc);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_PRE, mixed, il});
    } else {
        ggml_tensor * gated = ggml_mul(ctx0, xn, ggml_sigmoid(ctx0, gate));
        gated = ggml_reshape_3d(ctx0, gated, n_embd, hc, nt);

        // collapse the streams by their mean
        mixed = ggml_view_2d(ctx0, gated, n_embd, nt,
                ggml_row_size(gated->type, n_embd) * hc, 0);
        mixed = ggml_cont(ctx0, mixed);
        for (int64_t c = 1; c < hc; ++c) {
            ggml_tensor * s = ggml_view_2d(ctx0, gated, n_embd, nt,
                    ggml_row_size(gated->type, n_embd) * hc,
                    ggml_row_size(gated->type, n_embd) * c);
            mixed = ggml_add(ctx0, mixed, s);
        }
        mixed = ggml_scale(ctx0, mixed, 1.0f / (float) hc);
    }
    cb(mixed, "hc_mixed", il);

    if (inject) {
        *inject = di ? ggml_view_2d(ctx0, di, hc, nt, di->nb[1], ggml_row_size(di->type, di->ne[0] - hc))
                     : build_lora_mm(w_inject, xn);
        cb(*inject, "hc_inject", il);
    }

    return mixed;
}

ggml_tensor * llama_model_qwen4exp::graph::build_hc_combine(
        ggml_tensor * residual,
        ggml_tensor * block_out,
        ggml_tensor * inject,
        int           il) {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = residual->ne[2];

    // 2*sigmoid centres the scatter weights on 1, so a zero injection is a plain residual add
    ggml_tensor * w = ggml_sigmoid(ctx0, ggml_scale(ctx0, inject, 1.0f / (float) hc));
    w = ggml_scale(ctx0, w, 2.0f);

    ggml_tensor * cur = nullptr;
    if (cparams.fused_dsv4_hc_post && il >= 0) {
        // identity comb: every stream adds the same block output, scaled by its own weight
        cur = ggml_dsv4_hc_post(ctx0, block_out, residual, w, nullptr);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_POST, cur, il});
    } else {
        w = ggml_reshape_3d(ctx0, w, 1, hc, nt);

        ggml_tensor * b = ggml_reshape_3d(ctx0, block_out, n_embd, 1, nt);
        b = ggml_repeat_4d(ctx0, b, n_embd, hc, nt, 1);

        cur = ggml_add(ctx0, residual, ggml_mul(ctx0, b, w));
    }
    cb(cur, "hc_combine", il);

    return cur;
}

llama_model_qwen4exp::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t hc = hparams.dsv4_hc_mult;

    GGML_ASSERT(hparams.n_embd_head_v() == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.input_embed", -1);
    ggml_build_forward_expand(gf, inpL);

    auto * inp = build_inp_mem_hybrid();

    // qwen4exp always builds llama_memory_hybrid_idx, so this downcast is safe
    // the indexer cache inside it is absent when the GGUF has no indexer tensors
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(inp->mctx);

    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();
    if (mctx_idx) {
        GGML_ASSERT(mctx_idx->get_n_kv() == inp->mctx->get_attn()->get_n_kv() &&
                "the indexer cache must track the attention cache cell for cell");
    }

    // the QSA layers share one set of k-pool inputs
    // the CUDA lightning indexer takes 32 or 64 heads, QSA has a few, so it scores with plain ops
    llm_graph_input_kpool * inp_kpool = nullptr;
    if (mctx_idx && hparams.indexer_kpool > 0) {
        inp_kpool = build_inp_kpool(mctx_hyb);
    }

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * ple_emb = nullptr;
    if (hparams.ple_n_heads > 0) {
        ple_emb = build_inp_ple(mctx_hyb);
        // make sure ple_emb and build_inp_embd are in the same graph split
        ggml_build_forward_expand(gf, ple_emb);
    }

    // the wide residual starts as hc identical copies of the embedding
    ggml_tensor * res_hc = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(res_hc, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = res_hc;

        if (hparams.is_ple(il)) {
            res_hc = build_ple(inp->get_recr(), ple_emb, res_hc, il);
        }

        // MoE streaming hint (LLAMA_MOE_STREAM_HINT): this layer's router run on the layer input, before the attention,
        // through the FFN's own mixer. A host op starts the SSD reads of its top experts, which then overlap the
        // attention; the remap after the real router still decides which experts run.
        if (mstream && mstream->hint_k > 0 && n_tokens <= (int64_t) mstream->hint_max_tokens) {
            llama_moe_stream_layer * msl = mstream->layer(il);
            if (msl && msl->hint_store) {
                ggml_tensor * hx = build_hc_mix(res_hc,
                        model.layers[il].hc_ffn_norm,
                        model.layers[il].hc_ffn_down,
                        model.layers[il].hc_ffn_up,
                        model.layers[il].hc_ffn_inject,
                        nullptr, il, model.layers[il].hc_ffn_down_inject);
                ggml_tensor * hl = ggml_mul_mat(ctx0, model.layers[il].ffn_gate_inp, hx);
                ggml_prec_set_acc(hl, GGML_PREC_F32);
                cb(hl, "ffn_moe_hint_logits", il);
                // copied out of the compute buffer: the async op reads them after the GPU has gone on
                hl = ggml_cpy(ctx0, hl, ggml_view_2d(ctx0, msl->hint_store, hl->ne[0], hl->ne[1], msl->hint_store->nb[1], 0));
                ggml_tensor * hop = ggml_map_custom1(ctx0, hl, llama_moe_stream_hint, 1, msl);
                cb(hop, "ffn_moe_hint", il);
                ggml_build_forward_expand(gf, hop);
            }
        }

        ggml_tensor * inject = nullptr;
        ggml_tensor * cur = build_hc_mix(res_hc,
                model.layers[il].hc_attn_norm,
                model.layers[il].hc_attn_down,
                model.layers[il].hc_attn_up,
                model.layers[il].hc_attn_inject,
                &inject, il, model.layers[il].hc_attn_down_inject);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            if (!qwen4exp_skip("gdn")) {
                cur = build_layer_attn_linear(inp->get_recr(), cur, il);
            }
        } else {
            // the skip keeps the K/V and indexer cache writes (their inputs must stay in the graph),
            // the rest of the layer is pruned with its unused output
            ggml_tensor * attn_out = build_layer_attn(inp->get_attn(), mctx_hyb, inp_kpool, cur, inp_pos, sections, il);
            if (!qwen4exp_skip("attn")) {
                cur = attn_out;
            }
        }

        const bool gather_now = !cparams.embeddings_nextn || cparams.embeddings_nextn_masked;

        if (il == n_layer - 1 && inp_out_ids && gather_now) {
            // everything below is per token, so drop the rows that produce no output
            cur    = ggml_get_rows(ctx0, cur,    inp_out_ids);
            inject = ggml_get_rows(ctx0, inject, inp_out_ids);

            res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
            res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
            res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
        }

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        cur = build_hc_mix(res_hc,
                model.layers[il].hc_ffn_norm,
                model.layers[il].hc_ffn_down,
                model.layers[il].hc_ffn_up,
                model.layers[il].hc_ffn_inject,
                &inject, il, model.layers[il].hc_ffn_down_inject);

        cur = build_layer_ffn(cur, il);
        cb(cur, "ffn_out", il);

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        // "l_last" is the layer output name that build_cvec and imatrix look for
        cb(res_hc, "l_last", il);
    }

    // export res_hc itself, never a reshape view: a pure view gets no backend assignment to read back.
    if (cparams.embeddings_nextn) {
        cb(res_hc, "h_nextn", -1);
        res->t_h_nextn = res_hc;

        if (!cparams.embeddings_nextn_masked && inp_out_ids) {
            res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
            res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
            res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
        }
    }

    // the final mixer is the output norm: there is no separate one
    ggml_tensor * cur = build_hc_mix(res_hc,
            model.hc_head_norm, model.hc_head_down, model.hc_head_up,
            nullptr, nullptr, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// TODO: QSA for the draft head; dense is a numerical superset below the 2048-token budget.
llama_model_qwen4exp::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    graph(model, params, no_build_t{}) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "QWEN4EXP MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "QWEN4EXP MTP currently only supports a single MTP block");
    GGML_ASSERT(ubatch.token && "QWEN4EXP MTP requires token input");

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    GGML_ASSERT(hparams.n_embd_out() == (uint32_t) hc_dim && "QWEN4EXP MTP hidden width mismatch");

    const int il = hparams.n_layer() + cparams.nextn_layer_offset;
    GGML_ASSERT(cparams.nextn_layer_offset >= 0 &&
                cparams.nextn_layer_offset < (int) hparams.n_layer_nextn &&
                "nextn_layer_offset out of range [0, n_layer_nextn)");
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj     && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm       && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm       && "MTP block missing nextn.hnorm");
    GGML_ASSERT(layer.nextn.hc_head_norm && "MTP block missing nextn.hc_head_norm");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    auto inp = std::make_unique<llm_graph_input_embd_h>(hc_dim);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc_dim, n_tokens);
    ggml_set_input(inp->embd);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc_dim, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;
    if (tok_embd_w == nullptr) {
        tok_embd_w = qwen4exp_shared_model(cparams, model, "token_embd.weight").tok_embd;
        GGML_ASSERT(tok_embd_w && "QWEN4EXP MTP: the target model has no token embeddings to borrow");
    }
    ggml_tensor * tok_embd   = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    cb(tok_embd, "mtp_tok_embd", il);

    ggml_tensor * h_state = ggml_reshape_3d(ctx0, inp->h, n_embd, hc, n_tokens);
    cb(h_state, "mtp_h_state", il);

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * h_norm = ggml_rms_norm(ctx0, h_state, hparams.f_norm_rms_eps);
    h_norm = ggml_reshape_2d(ctx0, h_norm, hc_dim, n_tokens);
    h_norm = ggml_mul(ctx0, h_norm, layer.nextn.hnorm);
    h_norm = ggml_reshape_3d(ctx0, h_norm, n_embd, hc, n_tokens);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    e_norm = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, e_norm, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(e_norm, "mtp_enorm", il);

    // per stream, not pooled: pooling before the projection discards the hyper-connection residual.
    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * res_hc = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(res_hc, "mtp_eh_proj", il);

    ggml_tensor * inject = nullptr;
    ggml_tensor * cur = build_hc_mix(res_hc,
            layer.hc_attn_norm, layer.hc_attn_down, layer.hc_attn_up, layer.hc_attn_inject,
            &inject, il, layer.hc_attn_down_inject);
    cb(cur, "mtp_hc_attn_pre", il);

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * Qcur_full = build_lora_mm(layer.wq, cur, layer.wq_s);
    cb(Qcur_full, "mtp_Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "mtp_Qcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "mtp_gate", il);

    ggml_tensor * Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "mtp_Kcur_normed", il);

    ggml_tensor * Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    cb(Vcur, "mtp_Vcur", il);

    Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(Qcur, "mtp_Qcur", il);
    cb(Kcur, "mtp_Kcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f
            ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "mtp_attn_pregate", il);

    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
    cb(cur, "mtp_attn_gated", il);

    cur = build_lora_mm(layer.wo, cur, layer.wo_s);
    cb(cur, "mtp_attn_out", il);

    if (inp_out_ids) {
        cur    = ggml_get_rows(ctx0, cur,    inp_out_ids);
        inject = ggml_get_rows(ctx0, inject, inp_out_ids);

        res_hc = ggml_reshape_2d(ctx0, res_hc, hc_dim, res_hc->ne[2]);
        res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
        res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
    }

    res_hc = build_hc_combine(res_hc, cur, inject, il);
    cb(res_hc, "mtp_hc_attn_post", il);

    cur = build_hc_mix(res_hc,
            layer.hc_ffn_norm, layer.hc_ffn_down, layer.hc_ffn_up, layer.hc_ffn_inject,
            &inject, il, layer.hc_ffn_down_inject);
    cb(cur, "mtp_hc_ffn_pre", il);

    cur = build_layer_ffn(cur, il);
    cb(cur, "mtp_ffn_out", il);

    res_hc = build_hc_combine(res_hc, cur, inject, il);
    cb(res_hc, "mtp_hc_ffn_post", il);

    // the next draft step re-enters here, so export the wide stream before it is collapsed.
    cb(res_hc, "h_nextn", -1);
    res->t_h_nextn = res_hc;

    cur = build_hc_mix(res_hc,
            layer.nextn.hc_head_norm, layer.nextn.hc_head_down, layer.nextn.hc_head_up,
            nullptr, nullptr, -1);
    cb(cur, "mtp_hc_head", -1);

    // no res->t_embd: it is n_embd wide, but the context sizes that buffer by n_embd_out.

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    if (head_w == nullptr) {
        const llama_model & other = qwen4exp_shared_model(cparams, model, "output.weight");
        head_w = other.output;
        head_s = other.output_s;
        GGML_ASSERT(head_w && "QWEN4EXP MTP: the target model has no LM head to borrow");
    }

    if (layer.nextn.head_sub && layer.nextn.head_ids) {
        // reduced-vocabulary head: the K logits go to their token ids, every other token gets
        // -inf, so the sampler still sees a full-vocabulary row but can only draft from the subset
        const int64_t n_sub   = layer.nextn.head_sub->ne[1];
        const int64_t n_vocab = model.vocab.n_tokens();
        const int64_t n_out   = cur->ne[1];

        ggml_tensor * sub = build_lora_mm(layer.nextn.head_sub, cur, nullptr);
        cb(sub, "mtp_head_sub", -1);

        ggml_tensor * full = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_vocab, n_out), -INFINITY);
        full = ggml_set_rows(ctx0, full, ggml_reshape_3d(ctx0, sub, 1, n_sub, n_out), ggml_reshape_2d(ctx0, layer.nextn.head_ids, n_sub, 1));
        cur = ggml_reshape_2d(ctx0, full, n_vocab, n_out);
    } else {
        cur = build_lora_mm(head_w, cur, head_s);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen4exp::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen4exp::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    // the one numerical difference from Qwen3.5's GDN: sigmoid output gate, not silu
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated = ggml_sigmoid(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated);
}

// QSA k-pool inputs, shared by the QSA layers: blocks of compress_ratio cells in sequence order, see llama_memory_hybrid_idx
class llama_model_qwen4exp::llm_graph_input_kpool : public llm_graph_input_i {
public:
    llm_graph_input_kpool(const llama_memory_hybrid_idx_context * mctx, uint32_t kpool) : mctx(mctx), kpool(kpool) {}
    virtual ~llm_graph_input_kpool() = default;

    void set_input(const llama_ubatch * ubatch) override {
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);
        mctx->set_input_kpool(pool_cells, pool_idxs, pool_mask, tail_idxs, nullptr, false, new_pool_idxs, new_pool_rep,
                              ubatch, new_pool_pos, pool_lo, pool_hi);
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

        const auto * idx = mctx->get_idx();
        if (idx == nullptr) {
            return false;
        }

        bool res = true;

        res &= k_idxs->ne[0]     == params.ubatch.n_tokens;
        res &= pool_cells->ne[0] == mctx->get_n_kpool();
        res &= (pool_mask != nullptr ? pool_mask : pool_lo)->ne[1] == params.ubatch.n_tokens;
        res &= tail_idxs->ne[1]  == params.ubatch.n_tokens;
        // the scatter mask shape follows n_kv
        res &= n_kv              == idx->get_n_kv();
        res &= n_new             == mctx->get_n_kpool_new();
        res &= cache_safe        == mctx->get_kpool_cache_safe();

        return res;
    }

    ggml_tensor * k_idxs        = nullptr; // I64 [n_tokens]
    ggml_tensor * pool_cells    = nullptr; // I32 [n_pool]         cell caching each block's pooled key
    ggml_tensor * pool_idxs     = nullptr; // I32 [kpool, n_pool]  member cells per block, n_kv sentinel for the padded blocks
    ggml_tensor * pool_mask     = nullptr; // F32 or F16 [n_pool, n_tokens], absent when pool_lo and pool_hi are used
    ggml_tensor * pool_lo       = nullptr; // F32 [1, n_tokens]    first pool a token sees
    ggml_tensor * pool_hi       = nullptr; // F32 [1, n_tokens]    one past the last pool it sees
    ggml_tensor * tail_idxs     = nullptr; // I32 [kpool - 1, n_tokens]
    ggml_tensor * new_pool_idxs = nullptr; // I32 [kpool, n_new]   members of the blocks to re-pool this ubatch
    ggml_tensor * new_pool_rep  = nullptr; // I64 [n_new]          cell to write each new pooled key into
    ggml_tensor * new_pool_pos  = nullptr; // I32 [4*n_new]        M-RoPE position of each new block's first member

    // The rows [c0, c0 + tc) of the per-token inputs, one set of views per chunk for the whole graph: the scheduler copies
    // each view of an input to the GPU as a tensor of its own, so views made by every layer would be copied by every layer
    // (12 layers x the whole of kq_mask and pool_mask: 18 GiB at ctx 128k, ubatch 4096).
    struct chunk_views_t {
        ggml_tensor * kq_mask   = nullptr;
        ggml_tensor * pool_mask = nullptr;
        ggml_tensor * pool_lo   = nullptr;
        ggml_tensor * pool_hi   = nullptr;
        ggml_tensor * tail_idxs = nullptr;
    };
    std::map<std::pair<int64_t, int64_t>, chunk_views_t> chunks;

    const llama_memory_hybrid_idx_context * mctx;
    const uint32_t kpool;
    uint32_t n_new = 0; // padded to a stable bound, never below 1
    uint32_t n_sel = 0;
    uint32_t n_kv  = 0;
    bool cache_safe = true;
};

llama_model_qwen4exp::llm_graph_input_kpool * llama_model_qwen4exp::graph::build_inp_kpool(const llama_memory_hybrid_idx_context * mctx_hyb) {
    const auto * mctx_idx = mctx_hyb->get_idx();
    GGML_ASSERT(mctx_idx != nullptr);

    const uint32_t kpool  = hparams.indexer_kpool;
    const uint32_t n_pool = mctx_hyb->get_n_kpool();

    auto inp = std::make_unique<llm_graph_input_kpool>(mctx_hyb, kpool);

    inp->k_idxs     = mctx_idx->build_input_k_idxs(ctx0, ubatch);
    inp->pool_cells = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pool);
    inp->pool_idxs  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, kpool, n_pool);
    // NInfer-M1: the visibility of the blocks to a token is a prefix [lo, hi) of its sequence's pools. Past 16M elements
    // a [n_pool, n_tokens] input is too much (256 MiB as F16, on the host and again on the GPU, at ctx 128k and ubatch
    // 4096), so the graph builds the mask of a chunk from the two lengths per token. LLAMA_QSA_GPU_POOL_MASK=0 keeps the
    // input (F16 when large); both give the same mask.
    // LLAMA_QSA_POOL_MASK_ELEMS moves the 16M threshold (0: every graph builds it on the GPU, a huge value: none does)
    static const bool gpu_pool_mask = [] {
        const char * s = getenv("LLAMA_QSA_GPU_POOL_MASK");
        return s == nullptr || atoi(s) != 0;
    }();
    static const int64_t mask_elems = [] {
        const char * s = getenv("LLAMA_QSA_POOL_MASK_ELEMS");
        return s != nullptr ? (int64_t) atoll(s) : (int64_t) 1 << 24;
    }();
    const bool big_mask = (int64_t) n_pool*n_tokens >= mask_elems;
    if (big_mask && gpu_pool_mask) {
        inp->pool_lo = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
        inp->pool_hi = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
    } else {
        inp->pool_mask = ggml_new_tensor_2d(ctx0, big_mask ? GGML_TYPE_F16 : GGML_TYPE_F32, n_pool, n_tokens);
    }
    inp->tail_idxs  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, kpool - 1, n_tokens);
    ggml_set_input(inp->pool_cells);
    ggml_set_input(inp->pool_idxs);
    for (ggml_tensor * t : { inp->pool_mask, inp->pool_lo, inp->pool_hi }) {
        if (t != nullptr) {
            ggml_set_input(t);
        }
    }
    ggml_set_input(inp->tail_idxs);

    // set_input fills them all, so keep them allocated even when no op reads them
    ggml_build_forward_expand(gf, inp->pool_cells);
    ggml_build_forward_expand(gf, inp->pool_idxs);
    for (ggml_tensor * t : { inp->pool_mask, inp->pool_lo, inp->pool_hi }) {
        if (t != nullptr) {
            ggml_build_forward_expand(gf, t);
        }
    }
    ggml_build_forward_expand(gf, inp->tail_idxs);

    inp->n_kv       = mctx_idx->get_n_kv();
    inp->n_new      = mctx_hyb->get_n_kpool_new();
    inp->cache_safe = mctx_hyb->get_kpool_cache_safe();
    // the top blocks plus the tail
    inp->n_sel      = kpool*std::min<uint32_t>(n_pool, hparams.indexer_top_k / kpool) + kpool - 1;

    inp->new_pool_idxs = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, kpool, inp->n_new);
    ggml_set_input(inp->new_pool_idxs);
    if (inp->cache_safe) {
        inp->new_pool_rep = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, inp->n_new);
        ggml_set_input(inp->new_pool_rep);
    }
    inp->new_pool_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*inp->n_new);
    ggml_set_input(inp->new_pool_pos);

    return (llm_graph_input_kpool *) res->add_input(std::move(inp));
}

// QSA attends to the top blocks of compress_ratio cells plus the incomplete tail, like the glm5-next k-pool indexer
// a block is scored by one pooled key: the mean of its raw indexer keys, normed and rotated to its first member
//
// NInfer-M1: the selection mask is [n_kv, n_tokens] and its scoring temporaries are [n_pool, n_tokens]; at ctx 128k they
// took 3.75 GiB of compute buffer at ubatch 4096 (the Q4 then no longer fits the 50 GiB Metal limit). Every token is
// scored on its own, so build_qsa_mask works on a chunk of query tokens and build_attn_qsa runs the attention chunk by
// chunk: the allocator reuses the temporaries between chunks. LLAMA_QSA_CHUNK sets the chunk (default 256), 0 keeps one
// chunk (the graph of before); a ubatch of at most one chunk, decode included, builds the same graph as without chunks.
// Chunks of 256 tokens or more select the same blocks as one chunk, bit for bit (logprobs of 48 tokens after 5-6k tokens
// of context, ubatch 2048 and 4096); with 128 or 64 a few blocks at the edge of the top-k differ, so smaller values are
// raised to 256.
static int64_t qwen4exp_qsa_chunk() {
    static const int64_t chunk = [] {
        const char * s = getenv("LLAMA_QSA_CHUNK");
        const int64_t v = s != nullptr ? (int64_t) atoll(s) : (int64_t) 256;
        return v > 0 ? std::max<int64_t>(v, 256) : (int64_t) 0;
    }();
    return chunk;
}

bool llama_model_qwen4exp::graph::build_qsa_idx(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        llm_graph_input_kpool *                 inp_kpool,
        ggml_tensor *                           cur,
        ggml_tensor *                           inp_pos,
        int *                                   sections,
        int                                     il,
        qsa_idx &                               out) {
    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();

    const int64_t idx_dim = hparams.indexer_head_size;
    const int64_t n_idx_h = hparams.indexer_n_head;
    const int64_t kpool   = inp_kpool->kpool;
    const int64_t n_pool  = inp_kpool->pool_cells->ne[0];
    const int64_t n_new   = inp_kpool->n_new;

    GGML_ASSERT(hparams.dsv4_compress_ratios[il] == kpool);

    // cache rows store raw key | pooled key: pooling precedes norm and rotation, so the raw key gets neither
    ggml_tensor * k_raw = build_lora_mm(model.layers[il].index_k_proj, cur);
    cb(k_raw, "indexer_k_raw", il);

    ggml_tensor * pzero  = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, idx_dim, n_tokens), 0.0f);
    ggml_tensor * packed = ggml_reshape_3d(ctx0, ggml_concat(ctx0, k_raw, pzero, 0), 2*idx_dim, 1, n_tokens);
    ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, packed, inp_kpool->k_idxs, il));

    // the raw keys and the persistent pooled slots, see llama_memory_hybrid_idx::mem_idx_stale
    auto kpool_cache = mctx_hyb->get_kpool_access(ctx0, il, idx_dim);

    // pool only the blocks this ubatch completes or regroups
    ggml_tensor * rows = kpool_cache.gather_key_gate(ggml_reshape_1d(ctx0, inp_kpool->new_pool_idxs, kpool*n_new));
    rows = ggml_reshape_3d(ctx0, rows, idx_dim, kpool, n_new);

    // mean over the members; kpool is small, so summing slices beats a transpose plus sum_rows
    ggml_tensor * pooled_new = nullptr;
    for (int64_t i = 0; i < kpool; ++i) {
        ggml_tensor * slice = ggml_view_2d(ctx0, rows, idx_dim, n_new, rows->nb[2], i*rows->nb[1]);
        pooled_new = pooled_new ? ggml_add(ctx0, pooled_new, slice) : ggml_cont(ctx0, slice);
    }
    pooled_new = ggml_scale(ctx0, pooled_new, 1.0f/(float) kpool);
    pooled_new = build_norm(pooled_new, model.layers[il].index_k_norm, nullptr, LLM_NORM_RMS, il);

    pooled_new = ggml_reshape_3d(ctx0, pooled_new, idx_dim, 1, n_new);
    pooled_new = ggml_rope_multi(ctx0, pooled_new, inp_kpool->new_pool_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    pooled_new = ggml_reshape_2d(ctx0, pooled_new, idx_dim, n_new);
    cb(pooled_new, "indexer_pool_k_new", il);

    ggml_tensor * pooled = nullptr;
    if (inp_kpool->cache_safe) {
        // write before the pool gather
        ggml_build_forward_expand(gf, kpool_cache.scatter_pooled(pooled_new, inp_kpool->new_pool_rep));
        pooled = kpool_cache.gather_pooled(inp_kpool->pool_cells);
    } else {
        // shared cells re-pool every pool, in layout order
        GGML_ASSERT(n_new < n_pool);
        ggml_tensor * pad = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, idx_dim, n_pool - n_new), 0.0f);
        pooled = ggml_concat(ctx0, pooled_new, pad, 1);
    }
    // NInfer-M1: while every block fits the budget the selection keeps every visible cell, so it
    // equals the causal mask: the cache above is kept up to date and the layer attends densely.
    // This skips ~20 dispatches per QSA layer below indexer_top_k cells. LLAMA_QSA_DENSE_SHORT=0 disables it.
    static const bool dense_short = [] {
        const char * s = getenv("LLAMA_QSA_DENSE_SHORT");
        return s == nullptr || atoi(s) != 0;
    }();
    if (dense_short && n_pool <= (int64_t) (hparams.indexer_top_k / kpool)) {
        return false;
    }

    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, 1, n_pool);
    cb(pooled, "indexer_k", il);

    ggml_tensor * q = build_lora_mm(model.layers[il].index_q_proj, cur);
    q = ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h, n_tokens);
    q = build_norm(q, model.layers[il].index_q_norm, nullptr, LLM_NORM_RMS, il);
    q = ggml_rope_multi(ctx0, q, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q, "indexer_q", il);

    out.pooled = ggml_reshape_2d(ctx0, pooled, idx_dim, n_pool); // [idx_dim, n_pool]
    out.q      = q;                                               // [idx_dim, n_idx_h, n_tokens]
    return true;
}

// the additive mask [n_kv, tc] of the top blocks and the tail for the query tokens [c0, c0 + tc), kq_mask included
ggml_tensor * llama_model_qwen4exp::graph::build_qsa_mask(
        const qsa_idx &         qi,
        llm_graph_input_kpool * inp_kpool,
        ggml_tensor *           kq_mask,
        int64_t                 c0,
        int64_t                 tc,
        int                     il) {
    const int64_t idx_dim = hparams.indexer_head_size;
    const int64_t n_idx_h = hparams.indexer_n_head;
    const int64_t kpool   = inp_kpool->kpool;
    const int64_t n_pool  = inp_kpool->pool_cells->ne[0];
    const int64_t n_kv    = inp_kpool->n_kv;

    const bool whole = c0 == 0 && tc == n_tokens;

    // the query tokens [c0, c0 + tc) of the per-token inputs, shared by the layers
    ggml_tensor * pool_mask = inp_kpool->pool_mask;
    ggml_tensor * pool_lo   = inp_kpool->pool_lo;
    ggml_tensor * pool_hi   = inp_kpool->pool_hi;
    ggml_tensor * tail_idxs = inp_kpool->tail_idxs;
    ggml_tensor * mask      = kq_mask;
    if (!whole) {
        GGML_ASSERT(kq_mask->ne[2] == 1 && kq_mask->ne[3] == 1); // a chunk is a part of a single stream
        auto & cv = inp_kpool->chunks[{c0, tc}];
        if (cv.kq_mask == nullptr || cv.kq_mask->view_src != kq_mask) {
            auto tokens = [&](ggml_tensor * t) {
                return t == nullptr ? nullptr : ggml_view_2d(ctx0, t, t->ne[0], tc, t->nb[1], c0*t->nb[1]);
            };
            cv.pool_mask = tokens(pool_mask);
            cv.pool_lo   = tokens(pool_lo);
            cv.pool_hi   = tokens(pool_hi);
            cv.tail_idxs = tokens(tail_idxs);
            cv.kq_mask   = ggml_view_4d(ctx0, kq_mask, kq_mask->ne[0], tc, 1, 1,
                    kq_mask->nb[1], kq_mask->nb[2], kq_mask->nb[3], c0*kq_mask->nb[1]);
        }
        pool_mask = cv.pool_mask;
        pool_lo   = cv.pool_lo;
        pool_hi   = cv.pool_hi;
        tail_idxs = cv.tail_idxs;
        mask      = cv.kq_mask;
    }

    ggml_tensor * q = qi.q;
    if (!whole) {
        q = ggml_view_3d(ctx0, q, q->ne[0], q->ne[1], tc, q->nb[1], q->nb[2], c0*q->nb[2]);
    }

    // the reference sums the rectified head scores unweighted, scaled by 1/sqrt(head_dim)
    // one product for all heads, then the heads are summed as slices, so nothing is transposed
    ggml_tensor * kq = ggml_mul_mat(ctx0, qi.pooled,
            ggml_reshape_2d(ctx0, q, idx_dim, n_idx_h*tc)); // [n_pool, n_idx_h*tc]
    kq = ggml_relu(ctx0, ggml_reshape_3d(ctx0, kq, n_pool, n_idx_h, tc));

    ggml_tensor * score = nullptr;
    for (int64_t h = 0; h < n_idx_h; ++h) {
        ggml_tensor * slice = ggml_view_2d(ctx0, kq, n_pool, tc, kq->nb[2], h*kq->nb[1]);
        score = score ? ggml_add(ctx0, score, slice) : ggml_cont(ctx0, slice);
    }
    score = ggml_scale(ctx0, score, 1.0f/sqrtf((float) idx_dim));
    if (pool_mask != nullptr) {
        if (pool_mask->type != GGML_TYPE_F32) {
            pool_mask = ggml_cast(ctx0, pool_mask, GGML_TYPE_F32);
        }
        score = ggml_add(ctx0, score, pool_mask); // [n_pool, tc]
    } else {
        // the blocks [lo, hi) of a token are visible: 0 for them, -1e30 for the others. Block numbers and bounds are
        // integers far below 2^24, so the half-unit steps are exact and each step is 0 or 1. The top-k ties among the
        // blocks that are not visible, as it does with -inf, and nothing later reads their scores.
        ggml_tensor * block = ggml_repeat(ctx0,
                ggml_reshape_2d(ctx0, ggml_arange(ctx0, 0.0f, (float) n_pool, 1.0f), n_pool, 1), score); // [n_pool, tc]
        ggml_tensor * past_lo = ggml_step(ctx0, ggml_scale_bias(ctx0, ggml_sub(ctx0, block, pool_lo),  1.0f,  0.5f));
        ggml_tensor * before_hi = ggml_step(ctx0, ggml_scale_bias(ctx0, ggml_sub(ctx0, block, pool_hi), -1.0f, -0.5f));
        ggml_tensor * visible = ggml_mul(ctx0, past_lo, before_hi);
        score = ggml_add(ctx0, score, ggml_scale_bias(ctx0, visible, 1e30f, -1e30f)); // [n_pool, tc]
    }
    cb(score, "indexer_score", il);

    const int64_t n_top_pool = std::min<int64_t>(n_pool, hparams.indexer_top_k / kpool);
    ggml_tensor * top_k = ggml_top_k(ctx0, score, n_top_pool); // [n_top_pool, tc], unordered
    cb(top_k, "indexer_top_k", il);

    // the top blocks, then the incomplete tail with n_kv for missing cells
    ggml_tensor * sel_idx = ggml_get_rows(ctx0, inp_kpool->pool_idxs,
            ggml_reshape_1d(ctx0, top_k, n_top_pool*tc)); // [kpool, n_top_pool*tc]
    sel_idx = ggml_reshape_2d(ctx0, sel_idx, kpool*n_top_pool, tc);
    sel_idx = ggml_concat(ctx0, sel_idx, tail_idxs, 0);
    const int64_t n_sel = sel_idx->ne[0];
    GGML_ASSERT(n_sel == inp_kpool->n_sel);

    // scatter zeros for the selected cells into an all -inf row, the extra row n_kv takes the sentinels
    // seeding from sel_idx ties the scatter storage lifetime to this layer
    ggml_tensor * seed = ggml_cast(ctx0, ggml_view_1d(ctx0, sel_idx, 1, 0), GGML_TYPE_F32);

    ggml_tensor * mask_seed = kq_mask->type == GGML_TYPE_F32 ? seed : ggml_cast(ctx0, seed, kq_mask->type);
    mask_seed = ggml_fill(ctx0, mask_seed, -INFINITY);
    ggml_tensor * mask_all = ggml_repeat_4d(ctx0, mask_seed, 1, n_kv + 1, tc, 1);
    mask_all = ggml_reshape_3d(ctx0, mask_all, 1, n_kv + 1, tc);

    ggml_tensor * zero_seed = ggml_fill(ctx0, seed, 0.0f);
    ggml_tensor * zeros = ggml_repeat_4d(ctx0, zero_seed, 1, n_sel, tc, 1);
    zeros = ggml_reshape_3d(ctx0, zeros, 1, n_sel, tc);

    ggml_tensor * sel = ggml_set_rows(ctx0, mask_all, zeros, ggml_reshape_3d(ctx0, sel_idx, n_sel, tc, 1));

    GGML_ASSERT(mask->ne[0] == n_kv && mask->ne[1]*mask->ne[2]*mask->ne[3] == tc);
    const size_t row = sel->nb[2];
    sel = ggml_view_4d(ctx0, sel, n_kv, mask->ne[1], mask->ne[2], mask->ne[3],
            row, row*mask->ne[1], row*mask->ne[1]*mask->ne[2], 0);
    sel = ggml_add(ctx0, sel, mask);
    cb(sel, "indexer_sel", il);

    return sel;
}

// Dense GQA self-attention over the cells that the QSA mask keeps.
ggml_tensor * llama_model_qwen4exp::graph::build_attn_qsa(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             q_cur,
        ggml_tensor *             k_cur,
        ggml_tensor *             v_cur,
        const qsa_idx &           qi,
        llm_graph_input_kpool *   inp_kpool,
        float                     kq_scale,
        int                       il) {
    // rotate q/k/v before they reach a quantized cache, as the dense path does. the indexer
    // has already scored with its own query in build_qsa_idx, so the selection is unaffected.
    if (inp->self_k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }

    if (inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    ggml_tensor * kq_mask = inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    // the chunks need one stream; the selection mask already carries the causal mask
    int64_t chunk = qwen4exp_qsa_chunk();
    if (chunk <= 0 || chunk >= n_tokens || kq_mask->ne[2] != 1 || kq_mask->ne[3] != 1) {
        chunk = n_tokens;
    }

    ggml_tensor * cur = nullptr;
    for (int64_t c0 = 0; c0 < n_tokens; c0 += chunk) {
        const int64_t tc = std::min(chunk, n_tokens - c0);

        ggml_tensor * sel = build_qsa_mask(qi, inp_kpool, kq_mask, c0, tc, il);
        ggml_tensor * mask = ggml_reshape_4d(ctx0, sel, sel->ne[0], sel->ne[1], sel->ne[2], sel->ne[3]);
        cb(mask, "kq_mask_qsa", il);

        ggml_tensor * qc = tc == n_tokens ? q :
            ggml_view_3d(ctx0, q, q->ne[0], q->ne[1], tc, q->nb[1], q->nb[2], c0*q->nb[2]);

        ggml_tensor * out = build_attn_mha(qc, k, v, nullptr, mask, nullptr, nullptr, inp_kpool->n_sel, kq_scale, il);
        if (cur) {
            // the join goes into the graph now: the chunk's output holds the flash-attention scratch of its node
            // (K/V as F16, ~256 MiB at ctx 128k) until its consumer has run, so the chunks must not pile up
            cur = ggml_concat(ctx0, cur, out, 1);
            ggml_build_forward_expand(gf, cur);
        } else {
            cur = out;
        }
    }
    cb(cur, "kqv_out", il);

    // the rotation is its own inverse, so undo it on the value side of the output
    if (inp->self_v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
    }

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        const llama_memory_hybrid_idx_context * mctx_hyb,
        llm_graph_input_kpool *   inp_kpool,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // indexer reads the same block input as q/k/v; no cache or no ratio means dense
    const bool qsa = inp_kpool != nullptr && hparams.dsv4_compress_ratios[il] > 0;

    qsa_idx qi;
    const bool qsa_sel = qsa && build_qsa_idx(mctx_hyb, inp_kpool, cur, inp_pos, sections, il, qi);

    // Qwen3Next uses a single Q projection that outputs query + gate
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s); // [ (n_embd_head * 2) * n_head, n_tokens ]
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    cb(Vcur, "Vcur", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply IMRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    if (qsa_sel) {
        cur = build_attn_qsa(inp, Qcur, Kcur, Vcur, qi, inp_kpool, kq_scale, il);
    } else {
        cur = build_attn(inp,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    }
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = hparams.ssm_d_state;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);
    GGML_ASSERT(head_v_dim * num_v_heads == d_inner);

    ggml_tensor * qkv_mixed = nullptr;
    ggml_tensor * z         = nullptr;
    ggml_tensor * beta      = nullptr;
    ggml_tensor * alpha     = nullptr;
    if (model.layers[il].ssm_zba) {
        qkv_mixed = build_lora_mm(model.layers[il].wqkv, cur, model.layers[il].wqkv_s);
        qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
        cb(qkv_mixed, "linear_attn_qkv_mixed", il);

        // merged weight: one matmul, rows [z | beta | alpha]
        ggml_tensor * zba = build_lora_mm(model.layers[il].ssm_zba, cur);
        cb(zba, "zba", il);
        const size_t es = ggml_element_size(zba);
        z     = ggml_view_2d(ctx0, zba, d_inner, zba->ne[1], zba->nb[1], 0);
        cb(z, "z", il);
        beta  = ggml_view_4d(ctx0, zba, 1, num_v_heads, n_seq_tokens, n_seqs,
                es, zba->nb[1], zba->nb[1] * n_seq_tokens, d_inner * es);
        alpha = ggml_view_3d(ctx0, zba, num_v_heads, n_seq_tokens, n_seqs,
                zba->nb[1], zba->nb[1] * n_seq_tokens, (d_inner + num_v_heads) * es);
    } else {
        auto qkvz = build_qkvz(cur, il);
        qkv_mixed = qkvz.first;
        z         = qkvz.second;

        beta  = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
        beta  = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
        alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
        alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    }
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];

    // the channels must match how load_arch_tensors sizes wqkv, not ssm_d_inner
    const int64_t conv_channels    = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;

    ggml_tensor * conv_input = build_conv_state_at(inp, conv_states_all, qkv_mixed,
            conv_kernel_size - 1, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, conv_channels);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);

    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);

    // repeat to match shapes when head keys != value keys; unneeded with the fused GDN
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    ggml_tensor * z_2d = ggml_view_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(z->type, head_v_dim), z->nb[1], z->nb[1] * n_seq_tokens, 0);

    // gated normalization, as self.norm(core_attn_out, z) in the reference
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    ggml_tensor * moe_out = qwen4exp_skip("moe") ? cur :
        build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, model.layers[il].ffn_gate_up_exps,
            model.layers[il].ffn_up_exps_s,
            model.layers[il].ffn_gate_exps_s,
            model.layers[il].ffn_down_exps_s);
    cb(moe_out, "ffn_moe_out", il);

    // shared experts, as in the Qwen3Next reference
    if (!qwen4exp_skip("shexp") && (model.layers[il].ffn_up_shexp != nullptr || model.layers[il].ffn_gate_up_shexp != nullptr)) {
        // merged gate|up: one matmul, swiglu splits the halves
        ggml_tensor * ffn_shexp = model.layers[il].ffn_gate_up_shexp ?
            build_ffn(cur,
                model.layers[il].ffn_gate_up_shexp, NULL, NULL,
                NULL, NULL, NULL,
                model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                NULL,
                LLM_FFN_SWIGLU, LLM_FFN_SEQ, il) :
            build_ffn(cur,
                model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        // shared expert has its own sigmoided gate (ffn_gate_inp_shexp, one value per token)
        ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
        cb(shared_gate, "shared_expert_gate", il);

        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "shared_expert_gate_sigmoid", il);

        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "ffn_shexp_gated", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
    } else {
        cur = moe_out;
    }

    return cur;
}

// PLE n-gram hash embedding: each token gathers ple_n_heads rows of a shared table.
//   mixed_n = (t[p]*m[0]) ^ ... ^ (t[p-n+1]*m[n-1]);  row = mixed_n % vocab[h] + offset[h]
// The hash runs host-side because ggml has no int64 and no xor. EOS resets the window.

class llm_graph_input_ple : public llm_graph_input_i {
public:
    llm_graph_input_ple(const llama_model_qwen4exp & pmodel,
                        const llama_kv_cache_context * mctx) : pmodel(pmodel), mctx(mctx) {}
    virtual ~llm_graph_input_ple() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx)->get_attn();
        return rows->ne[0] == (int64_t) pmodel.hparams.ple_n_heads * params.ubatch.n_tokens;
    }

    ggml_tensor * rows = nullptr;   // I32 [ple_n_heads * n_tokens]

    const llama_model_qwen4exp & pmodel;

    // the predecessor tokens live in the attention KV cells (ext.tok)
    const llama_kv_cache_context * mctx;

    // scratch, reused across set_input() calls
    std::vector<llama_token> prev;
};

void llm_graph_input_ple::set_input(const llama_ubatch * ubatch) {
    const auto & hp = pmodel.hparams;

    // an image arrives as an embd batch, so ubatch->token is null, but every position still needs a row for ggml_get_rows
    // stand in the image token id that the reference hashes, or EOS if the file has no such key
    // gemma3n and gemma4 do the same with a hardcoded row 0 of per_layer_token_embd.
    const llama_token img_tok = hp.ple_image_token_id != 0
        ? (llama_token) hp.ple_image_token_id
        : (llama_token) hp.ple_eos_token_id;
    auto tok_of = [&](int64_t k) -> llama_token {
        return ubatch->token ? ubatch->token[k] : img_tok;
    };

    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_gram   = hp.ple_ngram_size;
    const int64_t n_heads  = hp.ple_n_heads;
    const int64_t per_gram = hp.ple_heads_per_ngram;
    const int64_t eos      = hp.ple_eos_token_id;
    const int64_t n_prev   = n_gram - 1;

    std::vector<int32_t> idx(n_heads * n_tokens);

    GGML_ASSERT(mctx != nullptr);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // the preceding tokens would be ambiguous, see get_prev_tokens()
        GGML_ASSERT(ubatch->n_seq_id[i] == 1 && "PLE n-gram embeddings do not support tokens shared by multiple sequences");
    }

    // predecessors come from the KV cells (ext.tok); apply_ubatch() already stored this ubatch, so its own tokens count too
    mctx->get_prev_tokens(*ubatch, n_prev, prev);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // an EOS in the window resets everything at or before it
        // a missing predecessor (before the sequence start, or no cached cell) reads as EOS
        // the EOS of the token itself does not cut its own context, as in the reference
        std::vector<int64_t> ctx(n_gram);
        ctx[0] = tok_of(i);
        bool cut = false;
        for (int64_t s = 1; s < n_gram; ++s) {
            // predecessor s positions back; prev[] is oldest-first, missing entries are LLAMA_TOKEN_NULL
            const llama_token t = cut ? LLAMA_TOKEN_NULL : prev[i*n_prev + (n_prev - s)];
            cut = cut || t < 0 || t == eos;
            ctx[s] = cut ? eos : t;
        }

        for (int64_t n = 2; n <= n_gram; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * hp.ple_layer_multipliers[0];
            for (int64_t j = 1; j < n; ++j) {
                mixed ^= (uint64_t) ctx[j] * hp.ple_layer_multipliers[j];
            }
            const int64_t base = (n - 2) * per_gram;
            for (int64_t g = 0; g < per_gram; ++g) {
                const int64_t h_i = base + g;
                idx[i * n_heads + h_i] =
                    (int32_t) (mixed % hp.ple_head_vocab_sizes[h_i] + hp.ple_head_offsets[h_i]);
            }
        }
    }

    ggml_backend_tensor_set(rows, idx.data(), 0, idx.size()*ggml_element_size(rows));

#ifdef QWEN4EXP_PLE_PREFETCH
    // The table is read lazily from the file mapping and the rows are scattered over ~29 GB, so
    // on new text almost every row is a page fault. The CPU get_rows would take them one by one
    // (~150 us each on an M1 Max SSD, ~9.5 ms for a 4-token verify); announcing all pages first
    // lets the kernel read them in parallel (~0.5 ms). LLAMA_PLE_PREFETCH=0 disables it.
    static const bool prefetch = [] {
        const char * s = getenv("LLAMA_PLE_PREFETCH");
        return s == nullptr || atoi(s) != 0;
    }();
    const ggml_tensor * table = pmodel.per_layer_tok_embd;
    if (prefetch && table && table->data && table->buffer && ggml_backend_buffer_is_host(table->buffer)) {
        const uintptr_t page = (uintptr_t) sysconf(_SC_PAGESIZE);
        const uintptr_t base = (uintptr_t) table->data;
        const size_t    nb1  = table->nb[1];
        for (const int32_t r : idx) {
            const uintptr_t beg = (base + (uintptr_t) r * nb1) & ~(page - 1);
            const uintptr_t end = (base + (uintptr_t) r * nb1 + nb1 + page - 1) & ~(page - 1);
            madvise((void *) beg, end - beg, MADV_WILLNEED);
        }
    }
#endif
}

// Read a conv history out of its own recurrent row and write the new tail back.
// The shared build_conv_state cannot do this: qwen4exp has two such rows per layer.
ggml_tensor * llama_model_qwen4exp::graph::build_conv_state_at(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_states_all,
        ggml_tensor *        x,
        int64_t              state_cols,
        int64_t              channels,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const auto kv_head = mctx_cur->get_head();

    const int64_t n_seqs    = ubatch.n_seqs;
    const int64_t row_total = conv_states_all->ne[0];

    // the row is exactly this convolution's state, so the gather is reused as a whole
    GGML_ASSERT(state_cols * channels == row_total);

    auto it = rs_rows.find(conv_states_all);
    if (it == rs_rows.end()) {
        it = rs_rows.emplace(conv_states_all, build_rs(inp, conv_states_all, row_total, n_seqs)).first;
    }
    ggml_tensor * rows = it->second;

    ggml_tensor * state = ggml_reshape_3d(ctx0, rows, state_cols, channels, n_seqs);
    cb(state, "conv_state_at", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, state, ggml_transpose(ctx0, x), 0);

    // [TAG_RECURRENT_ROLLBACK_SPLITS] keep the last state_cols columns once per rollback slot,
    // slot s ending s tokens earlier so a rollback of s tokens reads a history that never saw them
    const size_t row_size = ggml_row_size(conv_states_all->type, row_total);
    const uint32_t mem_size = mctx_cur->get_size();

    const int64_t n_slots = (int64_t) cparams.n_rs_seq + 1;

    for (int64_t slot = 0; slot < n_slots; ++slot) {
        const int64_t s_idx = std::max<int64_t>(0, conv_input->ne[0] - state_cols - slot);

        ggml_tensor * tail = ggml_view_3d(ctx0, conv_input,
                state_cols, channels, n_seqs,
                conv_input->nb[1], conv_input->nb[2],
                ggml_row_size(conv_input->type, s_idx));

        ggml_tensor * dst = ggml_view_2d(ctx0, conv_states_all,
                state_cols * channels, n_seqs,
                conv_states_all->nb[1],
                (slot * mem_size + kv_head) * row_size);

        // NInfer-M1: the copy kernels take strided sources, a ggml_cont here only adds a dispatch per slot and layer
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, tail, dst));
    }

    return conv_input;
}

ggml_tensor * llama_model_qwen4exp::graph::build_inp_ple(
        const llama_memory_hybrid_idx_context * mctx_hyb) {
    const int64_t n_heads = hparams.ple_n_heads;

    // the attention cells see every ubatch regardless of the layer types
    auto ple_inp = std::make_unique<llm_graph_input_ple>(
            static_cast<const llama_model_qwen4exp &>(model), mctx_hyb->get_attn());

    ple_inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_heads * n_tokens);
    ggml_set_input(ple_inp->rows);
    ggml_tensor * rows = ple_inp->rows;
    res->add_input(std::move(ple_inp));

    // gather then flatten the heads: get_rows lays the head dimension out slowest, as the reference does
    ggml_tensor * emb = ggml_get_rows(ctx0, model.per_layer_tok_embd, rows);
    emb = ggml_reshape_2d(ctx0, emb, hparams.ple_head_dim * n_heads, n_tokens);
    cb(emb, "ple_embd", -1);

    return emb;
}

ggml_tensor * llama_model_qwen4exp::graph::build_ple(
        llm_graph_input_rs * inp,
        ggml_tensor *        emb,
        ggml_tensor *        hidden,
        int                  il) {
    const int64_t hc      = hparams.dsv4_hc_mult;
    const int64_t hc_dim  = hc * n_embd;

    ggml_tensor * key   = build_lora_mm(model.layers[il].ple_key,   emb);
    ggml_tensor * value = build_lora_mm(model.layers[il].ple_value, emb);

    // both norms group over one hc stream, with a [n_embd, hc] weight
    auto grouped_norm = [&](ggml_tensor * x, ggml_tensor * w) {
        ggml_tensor * t = ggml_reshape_3d(ctx0, x, n_embd, hc, n_tokens);
        return ggml_mul(ctx0, ggml_rms_norm(ctx0, t, hparams.f_norm_rms_eps), w);
    };

    key = grouped_norm(key, model.layers[il].ple_norm_key);
    ggml_tensor * query = grouped_norm(hidden, model.layers[il].ple_norm_query);

    // per-stream dot product, then a signed square root before the sigmoid
    ggml_tensor * s = ggml_sum_rows(ctx0, ggml_mul(ctx0, key, query));
    s = ggml_scale(ctx0, s, 1.0f / sqrtf((float) n_embd));

    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, s), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, s), mag));
    cb(gate, "ple_gate", il);

    // [n_embd, 1, T] value broadcast across the hc streams, scaled by the gate
    ggml_tensor * v3 = ggml_reshape_3d(ctx0, value, n_embd, 1, n_tokens);
    v3 = ggml_repeat_4d(ctx0, v3, n_embd, hc, n_tokens, 1);

    ggml_tensor * gated = ggml_mul(ctx0, v3, gate);
    cb(gated, "ple_gated_value", il);

    ggml_tensor * normalized = grouped_norm(
            ggml_reshape_2d(ctx0, gated, hc_dim, n_tokens),
            model.layers[il].ple_norm_conv);
    normalized = ggml_reshape_2d(ctx0, normalized, hc_dim, n_tokens);

    // depthwise causal conv, dilated by the n-gram size, as a sum of shifted copies
    // ggml_conv_1d_dw is documented as unreliable:
    //   out[c, t] = sum_k w[k, c] * x[c, t - (K-1-k)*dilation]
    // The history of the earlier ubatches is prepended, so a chunked prefill matches a single-shot one.
    const int64_t kern = hparams.ple_conv_kernel;
    const int64_t dil  = hparams.ple_ngram_size;
    const int64_t hist = (kern - 1) * dil;

    // the conv history is per sequence, so the input carries the sequence axis too
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    // [hist + n_seq_tokens, hc_dim, n_seqs], tokens on ne[0]
    ggml_tensor * padded = build_conv_state_at(inp, inp->mctx->get_p_l(il),
            ggml_reshape_3d(ctx0, normalized, hc_dim, n_seq_tokens, n_seqs),
            hist, hc_dim, il);

    ggml_tensor * conv_out = nullptr;
    for (int64_t k = 0; k < kern; ++k) {
        // tap k reads (kern-1-k)*dilation positions back
        const int64_t start = hist - (kern - 1 - k) * dil;

        ggml_tensor * shifted = ggml_cont(ctx0,
                ggml_transpose(ctx0,
                        ggml_view_3d(ctx0, padded, n_seq_tokens, hc_dim, n_seqs,
                                padded->nb[1], padded->nb[2],
                                ggml_row_size(padded->type, start))));

        // column k of the [kern, hc_dim] kernel is one weight per channel
        ggml_tensor * wk = ggml_cont(ctx0,
                ggml_view_2d(ctx0, model.layers[il].ple_conv1d, 1, hc_dim,
                        model.layers[il].ple_conv1d->nb[1],
                        k * model.layers[il].ple_conv1d->nb[0]));
        // this kernel keeps the file type, so cast it before it multiplies an f32 activation
        wk = ggml_reshape_1d(ctx0, wk, hc_dim);
        if (wk->type != GGML_TYPE_F32) {
            wk = ggml_cast(ctx0, wk, GGML_TYPE_F32);
        }

        ggml_tensor * term = ggml_mul(ctx0, shifted, wk);
        conv_out = conv_out ? ggml_add(ctx0, conv_out, term) : term;
    }

    conv_out = ggml_silu(ctx0, conv_out);
    conv_out = ggml_reshape_3d(ctx0, ggml_cont(ctx0, conv_out), n_embd, hc, n_tokens);
    cb(conv_out, "ple_conv_out", il);

    return ggml_add(ctx0, hidden, ggml_add(ctx0, gated, conv_out));
}
