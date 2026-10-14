# Third-party notices

NInfer-M1 is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) and is released under the same MIT license
(see [`LICENSE`](../../LICENSE): Copyright (c) 2023-2026 The ggml authors). It builds on the work of others:

## llama.cpp and ggml

MIT. This repository keeps the full upstream history up to commit `9588757` (26 September 2026); everything after it is
the work of NInfer-M1 or an upstream change cherry-picked with `git cherry-pick -x`, which keeps its author and its upstream
reference (PR numbers in the commit subjects).

## MoE expert streaming

The streaming of routed experts from the SSD started as a port of the llama.cpp fork of
[mihailescu2m](https://github.com/mihailescu2m/llama.cpp), commit `d1762fc` (13 September 2026), MIT license. NInfer-M1
rebuilt most of the parts around it (host ops in the Metal command stream, Metal 4 submission, same-layer hints, the cache
policy and the memory limits); the port is commit `de69fe1`.

## Upstream changes ported by hand or cherry-picked

MTP drafting (llama.cpp PR #28243), the few-row matrix kernels of PRs #29869, #30065 and #30100, the qwen4exp mask
construction of #29824, and the fixes #29638, #29340, #29856, #29019, #30295 and #26004. Authors as in the upstream
repository.

## Model, weights and chat template

The model is [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by Qwen. Its weights are **not**
distributed here; the GGUF quantization by [mradermacher](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-i1-GGUF) and
the MTP draft by [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) are downloaded from their own
repositories, under their own terms.

`scripts/ninfer-m1/qwen3.8-flash-next.jinja` is the chat template of that model with one change (a system message in the
middle of a conversation is rendered instead of rejected). It is distributed under the license of the model, which requires
this notice to be included:

```
Qwen Community License 1.0

Copyright (c) 2026 Qwen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software, including the model weights, parameters, configuration files, inference code and associated documentation files (collectively, the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, sell, deploy, host, fine-tune, and create derivative works from (collectively, "Use" or "Using") copies of the Software; and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

1. The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. If the Software (or any derivative works thereof) is Used for any of the licensee's commercial products or services that have more than 100,000,000 monthly active users or US$ 20,000,000 (or equivalent in other currencies) monthly revenue, respective model name must be prominently displayed on the user interface of such product or service; and,

2. If the licensee or any of its affiliates conducts a Model as a Service or AI Work Assistant business, the licensee shall obtain a separate license from Qwen before Using the Software or its derivative works for any commercial purpose. The foregoing requirement shall not apply to the licensee's internal Use of the Software, provided that such Use does not make the Software, its outputs, or its underlying model capabilities available to any third party.

"Model as a Service" means giving a third party access to language model inference or fine-tuning (e.g., via API or a hosted endpoint) in a manner that allows such third parties to exercise meaningful control over the inputs, parameters, or training data. This does not include the mere relaying of requests to models hosted by other third parties. 
“AI Work Assistant” means an independent AI-powered product primarily designed for AI-assisted coding or office productivity (e.g., Qoder and QwenWork). It does not include: (a) a single-purpose AI tool (such as an AI translation tool); (b) an AI assistant primarily designed for a domain other than coding or office productivity (such as Taobao AI Shopping Assistant or AMap AI Chat); or (c) an AI assistant that is a feature of a product whose primary purpose is not AI-assisted coding or office productivity.

THE SOFTWARE AND ANY OUTPUT AND RESULTS THEREFROM ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL QWEN, ITS AFFILIATES OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. THE USE OF THE SOFTWARE MUST COMPLY WITH APPLICABLE LAWS AND REGULATIONS, AND MUST NOT INFRINGE THE INTELLECTUAL PROPERTY RIGHTS OF ANY THIRD PARTY.

For any questions regarding this license, please contact model-business@notice.qwencloud.com.
```

## Independent project

Not affiliated with ggml-org, the Qwen team, Apple, or the CUDA inference engine [ninfer](https://github.com/Neroued/ninfer)
by Neroued (Apache-2.0). The name echoes it: it was a design reference for the lab project this fork came out of. This
repository contains none of its code.
