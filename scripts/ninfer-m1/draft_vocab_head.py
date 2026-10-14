#!/usr/bin/env python3
"""Add a reduced-vocabulary LM head to a qwen4exp MTP draft GGUF (FR-Spec style).

Every draft step multiplies by the full LM head (248,320 rows, ~0.5 GB in Q6_K),
about half of its cost on the M1. The draft only needs to propose likely tokens:
the target model verifies every proposal with its full head, so output is
unchanged; a token outside the subset simply cannot be proposed.

Adds to the draft file:
  blk.N.nextn.head_sub.weight  [n_embd, K]  rows of the target's output.weight, copied byte for byte
  blk.N.nextn.head_ids.weight  [K] I32      token id of each row (ascending)

The patched qwen4exp MTP graph (this repository) multiplies by head_sub and
scatters the K logits into a -inf vector of the full vocabulary.

Subset: all non-normal tokens (control, user-defined: end of turn etc.), then
tokens by frequency in --counts (int32 token ids, e.g. from build/tokenize-ids),
then the lowest remaining ids until K.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_raw import build_header, check_contiguous, iter_chunks, read_header, verify_file, write_file  # noqa: E402

TYPE_I32 = 26
TOKEN_NORMAL = 1


def token_types(path):
    from gguf import GGUFReader
    field = GGUFReader(path).fields['tokenizer.ggml.token_type']
    return np.array([field.parts[i][0] for i in field.data], dtype=np.int32)


def select(n_vocab, k, types, counts):
    chosen = np.zeros(n_vocab, bool)
    chosen[np.nonzero(types != TOKEN_NORMAL)[0]] = True
    if counts is not None:
        freq = np.bincount(counts, minlength=n_vocab)[:n_vocab]
        order = np.lexsort((np.arange(n_vocab), -freq))
        for t in order[freq[order] > 0]:
            if chosen.sum() >= k:
                break
            chosen[t] = True
    for t in range(n_vocab):
        if chosen.sum() >= k:
            break
        chosen[t] = True
    return np.nonzero(chosen)[0].astype(np.int32)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('target', type=Path, help='target model GGUF (source of output.weight and token types)')
    ap.add_argument('draft', type=Path, help='MTP draft GGUF without head_sub')
    ap.add_argument('output', type=Path)
    ap.add_argument('--k', type=int, default=65536)
    ap.add_argument('--counts', type=Path, help='int32 token ids to rank tokens by frequency')
    ap.add_argument('--drop', help='regex of draft tensors to leave out, e.g. "^(token_embd|output)\\." to borrow them from the target')
    args = ap.parse_args(argv)
    if args.output.exists():
        ap.error(f'{args.output} exists')

    th = read_header(args.target)
    head = next(t for t in th['tensors'] if t['name'] == 'output.weight')
    n_embd, n_vocab = head['dims']
    row_bytes = head['nbytes'] // n_vocab
    types = token_types(args.target)
    if len(types) != n_vocab:
        ap.error(f'{len(types)} token types for {n_vocab} rows')
    counts = np.fromfile(args.counts, dtype=np.int32) if args.counts else None
    ids = select(n_vocab, args.k, types, counts)

    dh = read_header(args.draft)
    check_contiguous(dh)
    layers = sorted({int(t['name'].split('.')[1]) for t in dh['tensors'] if t['name'].startswith('blk.')})
    if len(layers) != 1:
        ap.error(f'expected one MTP block, found {layers}')
    il = layers[0]
    if any('.nextn.head_sub.' in t['name'] for t in dh['tensors']):
        ap.error('draft already has a reduced head')
    import re
    kept = [t for t in dh['tensors'] if not (args.drop and re.search(args.drop, t['name']))]
    tensors = [dict(t, new_dims=t['dims'], new_type=t['type'], kind='copy') for t in kept]
    tensors.append(dict(name=f'blk.{il}.nextn.head_sub.weight', new_dims=[n_embd, len(ids)], new_type=head['type'], kind='head'))
    tensors.append(dict(name=f'blk.{il}.nextn.head_ids.weight', new_dims=[len(ids)], new_type=TYPE_I32, kind='ids'))
    blob, tensors, _ = build_header(dh, tensors)

    partial = args.output.with_name(args.output.name + '.partial')
    with open(args.draft, 'rb') as fd, open(args.target, 'rb') as ft:
        def produce(t):
            if t['kind'] == 'copy':
                yield from iter_chunks(fd, dh['data_start'] + t['offset'], t['nbytes'])
            elif t['kind'] == 'ids':
                yield ids.tobytes()
            else:
                base = th['data_start'] + head['offset']
                for lo in range(0, len(ids), 4096):
                    part = []
                    for i in ids[lo:lo + 4096]:
                        ft.seek(base + int(i) * row_bytes)
                        part.append(ft.read(row_bytes))
                    yield b''.join(part)
        digests = write_file(partial, blob, tensors, dh['alignment'], produce)
    verify_file(partial, tensors, digests, dh['kv_raw'], dh['alignment'])
    partial.rename(args.output)

    coverage = None
    if counts is not None:
        mask = np.zeros(n_vocab, bool)
        mask[ids] = True
        coverage = float(mask[counts].mean())
    receipt = dict(target=str(args.target), draft=str(args.draft), output=str(args.output), k=len(ids),
                   n_vocab=n_vocab, head_type=head['type'], dropped=[t['name'] for t in dh['tensors'] if t not in kept], head_row_bytes=row_bytes,
                   non_normal_tokens=int((types != TOKEN_NORMAL).sum()), counts=str(args.counts) if args.counts else None,
                   counts_coverage=coverage, head_sub_bytes=len(ids) * row_bytes, verified=True)
    args.output.with_name(args.output.name + '.head.json').write_text(json.dumps(receipt, indent=1) + '\n')
    print(json.dumps(receipt, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
