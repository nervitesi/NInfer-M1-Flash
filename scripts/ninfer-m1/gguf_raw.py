"""Byte-level GGUF v3 reading and rewriting shared by the NInfer-M1 model tools.

The KV section is kept as raw bytes so metadata survives unchanged; scalar
values can be patched in place at their recorded offsets. Tensor data is
written contiguously in tensor-info order, each tensor padded to the file
alignment, which is what gguf.cpp (gguf_init_from_file_impl) requires.
"""
import hashlib
import os
import struct

from gguf import GGML_QUANT_SIZES

GGUF_MAGIC = b'GGUF'
GGUF_DEFAULT_ALIGNMENT = 32
CHUNK_BYTES = 8 * 1024 * 1024
TYPE_SIZES = {t.value: size for t, size in GGML_QUANT_SIZES.items()}
# GGUF metadata value types: fixed size in bytes; 8 = string, 9 = array
KV_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
KV_STRUCT = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d'}


class Reader:
    def __init__(self, f):
        self.f = f

    def take(self, n):
        data = self.f.read(n)
        if len(data) != n:
            raise ValueError('Truncated GGUF header')
        return data

    def u32(self):
        return struct.unpack('<I', self.take(4))[0]

    def u64(self):
        return struct.unpack('<Q', self.take(8))[0]

    def string(self):
        return self.take(self.u64()).decode('utf-8')

    def skip_value(self, vtype):
        if vtype in KV_FIXED:
            return self.take(KV_FIXED[vtype])
        if vtype == 8:
            return self.take(self.u64())
        if vtype == 9:
            item, count = self.u32(), self.u64()
            if item in KV_FIXED:
                self.take(KV_FIXED[item] * count)
            else:
                for _ in range(count):
                    self.skip_value(item)
            return None
        raise ValueError(f'Unknown GGUF value type {vtype}')


def tensor_nbytes(ggml_type, dims):
    if ggml_type not in TYPE_SIZES:
        raise ValueError(f'Unsupported ggml type {ggml_type}')
    block_elems, block_bytes = TYPE_SIZES[ggml_type]
    n = 1
    for d in dims:
        n *= d
    if dims[0] % block_elems:
        raise ValueError('Row size is not a multiple of the block size')
    return n // block_elems * block_bytes


def pad(n, alignment):
    return (n + alignment - 1) // alignment * alignment


def read_header(path):
    """Parse a GGUF v3 header. `scalars` maps key -> (value type, offset in kv_raw, value)."""
    with open(path, 'rb') as f:
        r = Reader(f)
        if r.take(4) != GGUF_MAGIC:
            raise ValueError('Not a GGUF file')
        version = r.u32()
        if version != 3:
            raise ValueError(f'Only GGUF v3 is supported, found v{version}')
        n_tensors, n_kv = r.u64(), r.u64()
        kv_start = f.tell()
        scalars = {}
        for _ in range(n_kv):
            key = r.string()
            vtype = r.u32()
            offset = f.tell() - kv_start
            raw = r.skip_value(vtype)
            if vtype in KV_STRUCT:
                scalars[key] = (vtype, offset, struct.unpack(KV_STRUCT[vtype], raw)[0])
        kv_end = f.tell()
        f.seek(kv_start)
        kv_raw = f.read(kv_end - kv_start)
        f.seek(kv_end)
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.u64() for _ in range(r.u32())]
            ggml_type = r.u32()
            offset = r.u64()
            tensors.append(dict(name=name, dims=dims, type=ggml_type, offset=offset, nbytes=tensor_nbytes(ggml_type, dims)))
        alignment = GGUF_DEFAULT_ALIGNMENT
        if 'general.alignment' in scalars:
            if scalars['general.alignment'][0] != 4:
                raise ValueError('general.alignment must be u32')
            alignment = scalars['general.alignment'][2]
        data_start = pad(f.tell(), alignment)
    if alignment == 0 or alignment & (alignment - 1):
        raise ValueError('Alignment must be a power of two')
    return dict(version=version, n_kv=n_kv, kv_raw=kv_raw, scalars=scalars, alignment=alignment,
                tensors=tensors, data_start=data_start)


def patch_scalar(header, key, value):
    """Return kv_raw with a scalar metadata value replaced by one of the same type."""
    vtype, offset, _ = header['scalars'][key]
    raw = bytearray(header['kv_raw'])
    packed = struct.pack(KV_STRUCT[vtype], value)
    raw[offset:offset + len(packed)] = packed
    return bytes(raw)


def check_contiguous(header):
    """Mirror gguf.cpp: offsets must follow tensor-info order, each padded to the alignment."""
    expected = 0
    for t in header['tensors']:
        if t['offset'] != expected:
            raise ValueError(f"Tensor {t['name']} has offset {t['offset']}, expected {expected}")
        expected += pad(t['nbytes'], header['alignment'])
    return expected


def encode_string(s):
    data = s.encode('utf-8')
    return struct.pack('<Q', len(data)) + data


def build_header(header, tensors, kv_raw=None):
    """tensors: dicts with name, dims (new), new_type; returns (header bytes, tensors with new_offset/new_nbytes, data size)."""
    out, offset = [], 0
    for t in tensors:
        nbytes = tensor_nbytes(t['new_type'], t['new_dims'])
        out.append(dict(t, new_offset=offset, new_nbytes=nbytes))
        offset += pad(nbytes, header['alignment'])
    blob = bytearray(GGUF_MAGIC + struct.pack('<IQQ', 3, len(out), header['n_kv']) + (kv_raw or header['kv_raw']))
    for t in out:
        blob += encode_string(t['name']) + struct.pack('<I', len(t['new_dims']))
        blob += b''.join(struct.pack('<Q', d) for d in t['new_dims'])
        blob += struct.pack('<IQ', t['new_type'], t['new_offset'])
    blob += b'\0' * (pad(len(blob), header['alignment']) - len(blob))
    return bytes(blob), out, offset


def iter_chunks(f, start, nbytes, step=CHUNK_BYTES):
    f.seek(start)
    left = nbytes
    while left:
        data = f.read(min(step, left))
        if not data:
            raise ValueError('Truncated tensor data')
        left -= len(data)
        yield data


def write_file(dst, blob, tensors, alignment, produce):
    """Write header then each tensor's bytes from produce(t) (an iterable of chunks); returns per-tensor sha256."""
    digests = {}
    with open(dst, 'wb') as fout:
        fout.write(blob)
        for t in tensors:
            h = hashlib.sha256()
            written = 0
            for chunk in produce(t):
                h.update(chunk)
                fout.write(chunk)
                written += len(chunk)
            if written != t['new_nbytes']:
                raise ValueError(f"{t['name']}: wrote {written} bytes, expected {t['new_nbytes']}")
            fout.write(b'\0' * (pad(t['new_nbytes'], alignment) - t['new_nbytes']))
            digests[t['name']] = h.hexdigest()
        fout.flush()
        os.fsync(fout.fileno())
    return digests


def verify_file(dst, tensors, digests, kv_raw, alignment):
    out = read_header(dst)
    check_contiguous(out)
    if out['kv_raw'] != kv_raw or out['alignment'] != alignment:
        raise ValueError('Metadata differs from what was written')
    if [(t['name'], t['dims'], t['type']) for t in out['tensors']] != [(t['name'], t['new_dims'], t['new_type']) for t in tensors]:
        raise ValueError('Tensor list differs from what was written')
    with open(dst, 'rb') as f:
        for got in out['tensors']:
            h = hashlib.sha256()
            for chunk in iter_chunks(f, out['data_start'] + got['offset'], got['nbytes']):
                h.update(chunk)
            if h.hexdigest() != digests[got['name']]:
                raise ValueError(f"Data mismatch after write: {got['name']}")
        f.seek(0, os.SEEK_END)
        if f.tell() != out['data_start'] + sum(pad(t['nbytes'], out['alignment']) for t in out['tensors']):
            raise ValueError('Unexpected output file size')
