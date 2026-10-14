# How NInfer-M1 runs a 97.5 GB model on a 64 GB Mac

This is the technical companion of the [README](../../README.md). Every number comes from a measurement on the
same machine — a 2021 MacBook Pro, **M1 Max, 32-core GPU, 64 GB unified memory, 1 TB internal SSD** — and is dated:
the engine changed a lot in three weeks.

## The model and the problem

**Qwen3.8-Flash-Next** is a hybrid mixture-of-experts model: 48 layers, **512 routed experts per layer, 10 active per
token** plus a shared expert, gated-delta-net (linear attention) layers mixed with full-attention layers that use a
query-selected sparse attention (QSA), hyper-connections between layers, a per-layer n-gram embedding table (51 B
parameters, read one row per token) and an MTP head for drafting. The model card gives 125 B parameters with 6 B activated
per token, plus the 51 B n-gram embedding and 4 B of MTP: about 180 B in all.

The file used here is the **mradermacher i1-IQ4_XS** GGUF, the closest thing to a "true Q4": 97.5 GB.

| Part | Bytes | How it is used |
|---|---:|---|
| routed experts (gate/up IQ4_XS, down IQ4_NL) | 65.4 GB | 10 of 512 per layer per token |
| per-layer n-gram table | 28.8 GB | one row per token and layer |
| everything else ("dense") | 3.2 GB | every token |

The Mac has 64 GB, and macOS lets the GPU wire roughly 50 GiB before things go wrong — on 2026-10-06 the kernel
panicked at 54.3 GiB of wired memory. So the experts cannot all live in memory. With a verification batch of 4 tokens
(3 drafted + 1), a step touches ~1,000 different experts, ~2.7 GB: the problem is to have the right ones in memory at
the right time, and to make the GPU work on them fast.

## 1. Experts streamed from the SSD

The streaming design comes from **mihailescu2m's llama.cpp fork (d1762fc, MIT)**, the first to run the full Flash Next on
a 64 GB Mac. NInfer-M1 ported it and rebuilt most of the parts around it.

- The dense weights stay resident. A **34 GiB Metal cache holds 286 of the 512 experts of each layer** (2.66 MB each).
- Right after the router of a layer, a *remap* runs on the CPU: missing experts are read from the GGUF with `pread`
  into free cache slots by 9 I/O threads, and the expert ids of the step are rewritten to slot ids. The router weights
  are untouched, so **the result is the same as with all experts in memory** (verified greedy, 15/15 identical texts).
- Eviction keeps the most used experts (a frequency halved every 1,024 steps). Simulated on a real routing trace of
  27k steps: 7.9% misses at 34 GiB; LRU 7.3%; the theoretical optimum (Belady) 3.8%, of which 2.2% are first reads.
- **Same-layer hints** (2026-10-11): before a layer's attention runs, the engine applies that layer's router to its
  input and starts the reads of each token's top-10 experts. The real router still decides; a wrong guess costs one
  read, never a different result. +2.6–3.2% on varied prompts, +6% when the cache is cold.
- Prefill: a ubatch of 4,096 tokens touches almost every expert; it runs in *waves* of experts that fit the cache.
- Measured with prompts that change from one request to the next, the GPU waits for the SSD **15–45%** of the
  generation time (`tools/bench_ssd_scenarios.py`). Repeating the same prompts hides it (3–4%).
- What did not help: `F_NOCACHE` reads (−4.9%), predicting 16 experts instead of 10 (the extra reads clog the SSD),
  a faster decay of the frequencies (+0.9% simulated, −2..−4% real), a 36 GiB cache (53.7 GiB wired: too close to the
  panic level).

## 2. The GPU side on an M1

- **Metal host ops and Metal 4 submission.** The remap is a CPU operation in the middle of each MoE layer. With the
  classic command queue every GPU→CPU→GPU round trip cost ~0.2 ms, 48 times per token. The fork runs registered CPU
  ops inside the command stream and submits these graphs through a Metal 4 queue (`GGML_METAL4=1`): the wait drops to
  ~0.05 ms (greedy 21.6 → 26.2 t/s without drafts, identical text). Two Metal 4 bugs found on the way: waits that
  failed behind ~3 s of GPU work, and buffers kept alive after being freed (2 GiB at load).
- **Few-row matrix products.** Verification multiplies every dense matrix by 4 rows. Upstream's few-row MMA kernel
  (simdgroup 8×8 matrices, each weight decoded once for all rows) was ported with **M1 Max thresholds, tilings and a
  large-batch window**: the dense kernels of a 4-row step went from 30.9 to 23.1 ms (−25%), 16 rows −74%; end to end
  +6..+10% with identical tokens.
- **Hyper-connection kernels**: M1-specific choices for the tall/thin HC projections (Q8_0 rows of 320 values, a few
  long rows): 2–4× faster per op, +16–18% decode on the in-memory model.
- Merged projections (`hc_*_down_inject`, `ssm_zba`, `ffn_gate_up_shexp`: 1,224 → 1,008 tensors) for the in-memory
  variant, fused `SCALE + UNARY + SCALE`, a parallel `mul_mm_id` map with atomic places and an indirect dispatch sized
  by it, an allocator fix (`ggml-alloc: check leaf sizes in needs_realloc`).

## 3. Attention: switching on the sparse attention the GGUF had lost

The Q4 GGUF stores a compress ratio of **0 on every layer**, so llama.cpp attended densely to the whole context. With
`LLAMA_QWEN4EXP_COMPRESS_RATIO=4` the full-attention layers get the ratio of the trained model: an indexer (4 heads of
128) scores pooled keys of 4 tokens and each query attends to its best 2,048 tokens plus the tail. At 30k tokens decode
went 17.5 → 29 t/s, at 65k 10.8 → 26.9 t/s (2026-10-13; 8 of 8 hidden codes retrieved in both).

Making it fit took three changes:

- **QSA by chunks of 256 query tokens**: the selection mask is `[n_kv, n_tokens]`; at 128k context and ubatch 4096 its
  temporaries took 3.75 GiB of compute buffer, and the model no longer loaded under the Metal limit. Chunked, the
  buffer grows by 33 MiB, and the logits are those of the unchunked graph (chunks are rounded to multiples of 256:
  other sizes moved a few logprobs by 1–2, cause not found).
- **The pool visibility mask is built on the GPU** from two lengths per token instead of a 256 MiB host input.
- **A deterministic top-k.** Above ~8k tokens the same prompt gave different logits from one run to the next: in 25 of
  172k rows two blocks tied for the last slot and the radix select took whichever thread came first. Ties now go to the
  lower block index; runs are bit-identical.
- 2026-10-14: the mask is built from one filled row repeated over the tokens (upstream #29824) instead of a one-element
  seed: prompt reading +8..+16% (210–217 t/s from 8k to 65k tokens).

## 4. Drafts

- **The MTP head** (from Unsloth's MTP GGUF) proposes 3 tokens, accepted when its probability is ≥ 0.6. Its LM head is
  replaced by a **64k-token sub-head** (FR-Spec style, `tools/draft_vocab_head.py`): rows of the target's own output
  matrix for the 65,536 most frequent tokens of a code corpus. The target verifies with its full head, so the output
  does not change; drafting gets cheaper (+3%).
- **N-gram drafts** of up to 32 tokens for edits that repeat existing text (+47% on agentic edits), **held back while the
  model is reasoning** (between `<think>` and `</think>`), where they were rejected and cost time.
- Verification is exact in the algorithm: a drafted token is kept only if the model itself picks it. With MTP drafts the
  greedy text matched plain decoding in 12 of 12 requests; with long n-gram drafts a bigger verification batch rounds
  differently and can flip a near-tie token after a few hundred tokens (BENCHMARKS, "Same output"). Exact rejection sampling for sampled
  requests is implemented (`LLAMA_SPEC_RS=1`, the same algorithm as upstream #27694) but off: at temperature 1.0 it was
  not faster (31.3 against 32.0 t/s), because the greedy draft is already close to optimal where the draft is unsure.
- The draft context now reads the prompt in micro-batches of 512 tokens (`LLAMA_SPEC_DRAFT_UBATCH`): its compute buffer
  went from 2,067 to 505 MiB.

## 5. Memory safety

A kernel panic on a laptop is worse than a slow answer, and this machine had two (2026-10-02 and 2026-10-06).

- The engine **refuses Metal allocations above physical memory − 14 GiB** (`GGML_METAL_MAX_ALLOC_GB`) instead of letting
  the GPU driver run out of wired memory.
- All benchmarks ran under a watchdog that stops the server at 53.7 GiB wired. It fired twice during this work, both
  times on configurations that are no longer used: the previous runtime at 65k tokens of context (54.0 GiB), and the
  in-memory AtomicChat variant at 131k context (54.3 GiB).
- With the final runtime the true Q4 peaks at **51.8–51.9 GiB at every context length up to 65k**.

## 6. Prompt cache for agent sessions

Claude Code sessions are long (median 57k tokens of context in two weeks of real use) and two of them often take turns.

- The prompt cache (states and context checkpoints) is written to disk at exit and restored at start: a server restart
  does not lose the conversation (`--prompt-cache-dir`).
- 2026-10-14: **a conversation that leaves the slot or the RAM cache goes to disk** (~1 GiB at 30k tokens, written in
  ~1.3 s) and is loaded back in ~0.3 s when a request continues it. Two 30k conversations taking turns: the answer to the
  first one came back after **5.6 s instead of 140 s**, identical to the same request read from scratch.

## What did not work

- A grouped MoE kernel for the verification batch: slower than the per-expert path on the M1.
- The CPU as a second matrix engine: the GPU alone reaches 348 GB/s, GPU + CPU 378 GB/s, not worth the
  synchronization.
- Two slots: no session gets faster (each loses 35% of generation speed); total throughput +30%.
- Converting all dense weights to IQ4_NL (−15% on the 4-token verification).
- DeepSeek V4.1 Flash on this Mac: 35 t/s prompt reading at 14k tokens, not competitive.
- Upstream's MoE/SSM_CONV fusions and FA fixes were already in the fork's base; their gains are already counted.
