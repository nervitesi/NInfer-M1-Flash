# Three weeks, one laptop: the log of NInfer-M1

The goal set at the start (September 2026): **Qwen3.8-Flash-Next at the best possible speed on a 64 GB M1 Max, with
the quality of the model preserved** — speed bought with a worse model did not count. Everything below was measured
on the same machine; the full lab notebook (in Italian, ~3,400 lines, every failure included) stays in the private lab
repository, and this page is its summary.

| Date | Step | Result |
|---|---|---|
| Sep 26–27 | Roofline of the M1 Max for this model; reference runs of llama.cpp, MLX engines and the streaming fork of mihailescu2m | the GPU reaches 348 GB/s; a 95 GB Q4 does not fit in memory, the streaming fork serves it at 10–13 t/s |
| Sep 28–29 | Merged projections (1,224 → 1,008 tensors); M1 kernel tuning | +10% and +9%, identical text |
| Sep 29 | 64k-token draft head (FR-Spec style), per-layer embedding prefetch | +3%, +6.6% on cold requests |
| Oct 1–2 | MTP drafts from upstream PR 28243, long-context tests, prompt cache persisted across restarts | a long conversation survives a server restart |
| Oct 2, Oct 6 | Two kernel panics while benchmarking | memory guard, watchdog, Metal allocation limit |
| Oct 2–3 | The engine replaces the old fork behind a local Claude Code router | daily use from here on |
| Oct 5 | N-gram drafts held back while the model reasons; exact rejection sampling (off: not faster) | +47% on agentic edits, +25% with reasoning on |
| Oct 6 | Hyper-connection kernels for the M1; expert streaming ported from mihailescu2m; Metal 4 submission for host ops | +16–18%; the true Q4 at 32.8 / 34.2 / 32.9 t/s (Italian / code / math) against 21.7 / 20.8 / 19.8 with the original fork |
| Oct 7, Oct 11 | Two Metal 4 bugs: waits that failed behind ~3 s of GPU work, a 2 GiB buffer leak at load | fixed |
| Oct 11 | Same-layer expert hints; routing trace and cache-policy simulation | +2.6–3.2%, +6% cold |
| Oct 13 | Few-row matrix products (upstream kernel + M1 windows) | dense kernels of the 4-token verification −25%, end to end +6..+10% |
| Oct 13 | Sparse attention switched on for the true Q4 (its GGUF had lost the compress ratios); chunked QSA graph; deterministic top-k | decode at 30k 17.5 → 29 t/s, at 65k 10.8 → 26.9 t/s |
| Oct 13 | Survey: upstream llama.cpp changes, new quantizations of the model, 2026 papers on SSD-offloaded MoE | the large lossless gains were already taken; the big wins in the papers change the output and were left out |
| Oct 14 | QSA mask from upstream, draft micro-batch, six upstream fixes, prompt cache on disk for alternating conversations | prompt reading +8..+16%, decode +3..+8%, 2 GiB less wired memory; back to a 30k conversation in 5.6 s instead of 140 s |
| Oct 14 | Final comparison with the other engines on the same file, then this release | 30.9 tok/s against 16.0 (streaming fork) and 9.7 (llama.cpp master); 29.1 against 13.2 and 4.8 after a 28k-token prompt; same text |

## Real use

From the router's own logs (timings only, no text), two weeks of Claude Code on this Mac: 2,298 requests, 83% of the
server time spent generating, a median context of 57k tokens at the end of a request (p90 92k), 84% of the generation
time above 32k tokens of context. That is why the sparse attention and the prompt cache mattered more than any kernel.

## The rule that kept the quality

Every change was checked against its own reference: greedy texts identical character by character, logprobs identical
to the last digit where the numerics should not change, hidden codes retrieved from long contexts, the graded
12-task suite. Two changes that made the output non-reproducible were found this way and fixed (the top-k ties, and
QSA chunk sizes that are not a multiple of 256).
