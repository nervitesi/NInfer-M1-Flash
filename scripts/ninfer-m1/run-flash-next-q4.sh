#!/bin/zsh
# Qwen3.8-Flash-Next at true Q4 (i1-IQ4_XS, 97.5 GB) on a 64 GB Apple Silicon Mac with NInfer-M1.
# An OpenAI- and Anthropic-compatible server on 127.0.0.1:${PORT:-8080}.
#
# What it does: the dense weights stay in memory (~3.3 GB), a 34 GiB Metal cache keeps the most used routed experts
# (286 of 512 per layer), the rest is read from the SSD when the router picks it; the 28.8 GB per-layer embedding table
# is read row by row from the file. Drafts: the MTP head (3 tokens) and n-gram drafts. The full-attention layers run
# the query-selected sparse attention (QSA) of the trained model.
#
# Environment:
#   MODEL        Qwen3.8-Flash-Next.i1-IQ4_XS.gguf (mradermacher/Qwen3.8-Flash-Next-i1-GGUF), required
#   DRAFT        the MTP draft GGUF (unsloth/Qwen3.8-Flash-Next-GGUF, MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf);
#                empty = no MTP drafts, n-gram drafts only (22.8 instead of 30.9 tok/s on short prompts)
#   NINFER       folder with llama-server (default: next to this script, else build/bin of this repository)
#   TEMPLATE     chat template (default: qwen3.8-flash-next.jinja next to this script: the model's own template with one
#                change, a system message in the middle of a conversation is rendered instead of rejected; agents send
#                them). TEMPLATE= (empty) uses the template inside the GGUF.
#   NGRAM        1 (default): n-gram drafts next to the MTP ones, they copy text that already appeared and speed up edits;
#                0: MTP drafts only, the configuration of the published benchmarks (n-gram drafts replay a prompt that is
#                repeated, which inflates a benchmark that repeats its prompts)
#   CTX          context length (default 131072; the KV cache is q8_0)
#   CACHE_GIB    expert cache (default 34). Do not raise it on a 64 GB Mac: at 36 GiB the wired memory reached 53.7 GiB
#                and the kernel panicked at 54.3 GiB in an earlier test with another model
#   PORT         default 8080
#   PROMPT_CACHE folder for the prompt cache on disk (default ~/.cache/ninfer-m1/prompt-cache); conversations that leave
#                the RAM cache are written there and loaded back (budget 24 GiB)
set -euo pipefail

here=${0:A:h}
if [[ -z ${NINFER:-} ]]; then
  if [[ -x $here/llama-server ]]; then NINFER=$here; else NINFER=$here/../../build/bin; fi
fi
[[ -x $NINFER/llama-server ]] || { print -u2 "llama-server not found in $NINFER (set NINFER, or build it: see README)"; exit 1; }
MODEL=${MODEL:?set MODEL to the Qwen3.8-Flash-Next i1-IQ4_XS GGUF}
[[ -f $MODEL ]] || { print -u2 "model file not found: $MODEL"; exit 1; }
DRAFT=${DRAFT:-}
NGRAM=${NGRAM:-1}
CTX=${CTX:-131072}
CACHE_GIB=${CACHE_GIB:-34}
PORT=${PORT:-8080}
PROMPT_CACHE=${PROMPT_CACHE:-$HOME/.cache/ninfer-m1/prompt-cache}
mkdir -p $PROMPT_CACHE

# Metal 4 submission for the graphs with host ops (the streamed MoE layers): the per-layer wait drops ~4x.
# Needs macOS 26; the engine ignores it with a warning on older systems.
export GGML_METAL4=${GGML_METAL4:-1}
# the layer's router runs before its attention and starts the SSD reads of each token's top-10 experts early
export LLAMA_MOE_STREAM_HINT=${LLAMA_MOE_STREAM_HINT:-10}
# the GGUF stores compress ratio 0 on every layer: give the full-attention layers the ratio of the trained model (QSA)
export LLAMA_QWEN4EXP_COMPRESS_RATIO=${LLAMA_QWEN4EXP_COMPRESS_RATIO:-4}
# the engine refuses Metal allocations above physical memory - 14 GiB (GGML_METAL_MAX_ALLOC_GB overrides it) instead of
# letting the GPU driver run out of wired memory; do not raise it

template_args=()
if [[ ${TEMPLATE-$here/qwen3.8-flash-next.jinja} != "" ]]; then
  template_args=(--chat-template-file ${TEMPLATE-$here/qwen3.8-flash-next.jinja})
fi

draft_args=()
ngram_args=(--spec-ngram-mod-n-min 16 --spec-ngram-mod-n-max 32)
if [[ -n $DRAFT && $NGRAM != 0 ]]; then
  draft_args=(--model-draft $DRAFT --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --spec-draft-p-min 0.6 -ngld 999 $ngram_args)
elif [[ -n $DRAFT ]]; then
  draft_args=(--model-draft $DRAFT --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.6 -ngld 999)
elif [[ $NGRAM != 0 ]]; then
  draft_args=(--spec-type ngram-mod $ngram_args)
fi

exec $NINFER/llama-server \
  --model $MODEL \
  --alias qwen3.8-flash-next-q4 \
  --host 127.0.0.1 --port $PORT \
  --ctx-size $CTX --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on \
  --n-predict -1 --gpu-layers all --fit off --load-mode dio --lazy-mode on \
  --jinja $template_args \
  --parallel 1 --threads 1 --threads-batch 8 \
  --batch-size 4096 --ubatch-size 4096 \
  --moe-stream-cache $CACHE_GIB \
  $draft_args \
  --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0 \
  --reasoning auto --reasoning-effort xhigh --reasoning-budget -1 --reasoning-preserve \
  --cache-prompt --cache-ram 2048 --ctx-checkpoints 8 --prompt-cache-dir $PROMPT_CACHE \
  --metrics --timeout 3600 --sse-ping-interval 15 --log-colors off
