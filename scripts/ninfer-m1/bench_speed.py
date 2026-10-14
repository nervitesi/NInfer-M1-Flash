#!/usr/bin/env python3
"""Measure the speed of any OpenAI-compatible chat server the way docs/ninfer-m1/BENCHMARKS.md does.

Start your engine first (NInfer-M1 with NGRAM=0, llama.cpp, MTPLX, mlx-serve, ...), then:

  bench_speed.py --port 8080                        3 short prompts x (1 warm-up + 3 timed requests), 256 new tokens
  bench_speed.py --port 8080 --long-tokens 28000    one long request: ~28k tokens of source code, then a question

Requests are greedy and streamed, thinking is off, the prompt cache is off. Generation speed is measured on the client:
(tokens - 1) / time between the first and the last streamed chunk, so it leaves the prompt out. For the long request the
wait for the first token and the prompt reading speed are reported too. Only the Python standard library is needed.

The long prompt is made of the *.cpp / *.h / *.py files under --root (default: the repository this script is in), so its
text is not the one of the published table; the protocol is. Do not run it against a server that also serves other
requests: the numbers would not mean anything.
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

PROMPTS = {
    'italian': 'Spiega in italiano come funziona un motore elettrico, con un esempio pratico.',
    'code': 'Implement a Python function merge_sorted(a, b) that merges two sorted lists in O(n+m) time. '
            'Include two tests and a short explanation.',
    'math': 'Solve x^2 - 5x + 6 = 0 in two different ways and verify each root. Explain the reasoning step by step.',
}
QUESTION = ('Using the code above as context, write a Python function `summarize_rows(path)` that reads a JSONL file '
            'of benchmark rows and returns, per engine and case, the median decode_tps. Include a docstring and two unit tests.')
CHARS_PER_TOKEN = 3.3  # rough for source code; the server reports the real prompt size


def chat(base, model, messages, max_tokens, timeout):
    body = {'model': model, 'messages': messages, 'max_tokens': max_tokens, 'temperature': 0, 'top_k': 1, 'seed': 42,
            'stream': True, 'stream_options': {'include_usage': True},
            'chat_template_kwargs': {'enable_thinking': False}, 'enable_thinking': False, 'cache_prompt': False}
    req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    start = time.monotonic()
    first = last = None
    chunks, usage, timings = 0, {}, {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            if not raw.startswith(b'data: '):
                continue
            line = raw[6:].strip()
            if line == b'[DONE]':
                break
            item = json.loads(line)
            if 'error' in item:
                raise RuntimeError(str(item['error']))
            usage = item.get('usage') or usage
            timings = item.get('timings') or timings
            for choice in item.get('choices', []):
                delta = choice.get('delta', {})
                if (delta.get('content') or '') + (delta.get('reasoning_content') or ''):
                    now = time.monotonic()
                    first = first or now
                    last = now
                    chunks += 1
    n = usage.get('completion_tokens') or timings.get('predicted_n') or chunks
    decode = (n - 1) / (last - first) if n > 1 and first and last and last > first else None
    return {'wait_s': (first - start) if first else None, 'tokens': n, 'decode_tps': decode,
            'prompt_tokens': usage.get('prompt_tokens') or timings.get('prompt_n'), 'timings': timings}


def count_tokens(base, text):
    """The server's own token count (llama-server /tokenize); None when the server has no such endpoint."""
    try:
        req = urllib.request.Request(base + '/tokenize', data=json.dumps({'content': text}).encode(),
                                     headers={'Content-Type': 'application/json'})
        return len(json.load(urllib.request.urlopen(req, timeout=120))['tokens'])
    except Exception:
        return None


def long_prompt(base, root, n_tokens):
    files = sorted(p for p in root.rglob('*') if p.suffix in ('.cpp', '.h', '.py') and p.is_file() and p.stat().st_size < 400_000
                   and 'build' not in p.parts and '.git' not in p.parts)
    parts, size = [], 0
    for p in files:
        text = f'### File: {p.relative_to(root)}\n```\n{p.read_text(errors="replace")}\n```\n'
        parts.append(text)
        size += len(text)
    body = ''.join(parts)
    target = int(n_tokens * CHARS_PER_TOKEN)
    for _ in range(2):  # one correction with the server's own count
        if len(body) < target:
            sys.exit(f'only {len(body)} characters of source under {root}, {target} needed: pass --root')
        prompt = 'You are a coding assistant. The repository files follow.\n\n' + body[:target]
        counted = count_tokens(base, prompt)
        if counted is None or abs(counted - n_tokens) < 0.03 * n_tokens:
            break
        target = int(target * n_tokens / counted)
    return prompt


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--model', default=None, help='model id (default: the first one the server lists)')
    ap.add_argument('--max-tokens', type=int, default=256)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--long-tokens', type=int, default=0, help='run one long request of about this many prompt tokens instead')
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument('--timeout', type=int, default=7200)
    args = ap.parse_args()
    base = f'http://{args.host}:{args.port}'
    model = args.model or json.load(urllib.request.urlopen(base + '/v1/models', timeout=10))['data'][0]['id']
    print(f'server {base}, model {model}')

    if args.long_tokens:
        messages = [{'role': 'system', 'content': long_prompt(base, args.root, args.long_tokens)}, {'role': 'user', 'content': QUESTION}]
        r = chat(base, model, messages, 400, args.timeout)
        read = r['prompt_tokens'] / r['wait_s'] if r['prompt_tokens'] and r['wait_s'] else None
        print(json.dumps({'prompt_tokens': r['prompt_tokens'], 'wait_for_first_token_s': round(r['wait_s'], 1),
                          'prompt_reading_tps': round(read, 1) if read else None,
                          'generation_tps': round(r['decode_tps'], 2) if r['decode_tps'] else None, 'tokens': r['tokens']}))
        return

    timed = []
    print(f"{'prompt':8s} {'first request':>14s}   timed requests")
    for name, prompt in PROMPTS.items():
        messages = [{'role': 'user', 'content': prompt}]
        warm = chat(base, model, messages, args.max_tokens, args.timeout)
        runs = [chat(base, model, messages, args.max_tokens, args.timeout)['decode_tps'] for _ in range(args.repeats)]
        timed += runs
        print(f"{name:8s} {warm['decode_tps']:14.1f}   " + '  '.join(f'{x:.1f}' for x in runs))
    print(f'median of {len(timed)} timed requests: {statistics.median(timed):.1f} tokens/s  (min {min(timed):.1f}, max {max(timed):.1f})')
    print('Note: n-gram drafts replay a prompt that is repeated, so a server that uses them reports inflated speeds here;\n'
          'for NInfer-M1 start the launcher with NGRAM=0 (MTP drafts only) to get the like-for-like figure.')


if __name__ == '__main__':
    main()
