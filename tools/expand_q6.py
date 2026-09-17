#!/usr/bin/env python3
"""Rearrange a packed LFM Q6 model losslessly for the Pentium 4 SSE2 kernel.

Requires NumPy on the packaging host. No conversion or extra allocation is
needed on XP: the resulting MODEL.NCB is mapped directly from disk.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def expand(source, output):
    source, output = Path(source), Path(output)
    with source.open('rb') as src:
        def read(size):
            block = src.read(size)
            if len(block) != size:
                raise ValueError('Truncated model tensor')
            return block

        header = bytearray(read(256))
        fields = struct.unpack('<8s14I', header[:64])
        magic, version, bits, d, ff, nl, nh, nk, hd, vocab, ctx, group, bos, eos, arch = fields
        theta, eps, conv, start = struct.unpack_from('<ffII', header, 64)
        if (magic != b'SLMODEL1' or version != 2 or bits != 6 or arch != 1 or group != 64
                or not 64 <= d <= 4096 or d % 64 or not 64 <= ff <= 16384 or ff % 64
                or not 1 <= nl <= 128 or not 1 <= nk <= nh <= 64 or nh % nk
                or not 2 <= hd <= 256 or hd % 2 or nh * hd != d or nk * hd % 4
                or not 256 <= vocab <= 200000 or vocab % 4 or not 32 <= ctx <= 8192
                or max(bos, eos, start) >= vocab or conv != 3
                or not np.isfinite(theta) or theta <= 1 or not np.isfinite(eps) or eps <= 0
                or any(t > 1 for t in header[80:80 + nl])):
            raise ValueError('Expected a supported version-2 LFM Q6/group64 model')
        struct.pack_into('<I', header, 8, 3)
        struct.pack_into('<I', header, 12, 8)  # storage width; learned precision remains Q6
        # Exclusive creation protects the source and any existing output.
        out = output.open('xb')
        try:
            with out:
                out.write(header)

                def copy(size):
                    while size:
                        n = min(size, 1048576)
                        out.write(read(n))
                        size -= n

                def matrix(rows, cols):
                    groups = cols // 64
                    # At most 64 rows in memory; reshape changes layout, not values.
                    for row in range(0, rows, 64):
                        count = min(rows - row, 64)
                        packed = np.frombuffer(read(count * groups * 48), np.uint8).reshape(count, groups, 48)
                        values = np.empty((count, groups, 64), np.uint8)
                        repacked = np.zeros_like(packed)
                        for j in range(64):
                            values[:, :, j] = ((packed[:, :, j // 2] >> (4 * (j % 2))) & 15) | (((packed[:, :, 32 + j // 4] >> (2 * (j % 4))) & 3) << 4)
                            repacked[:, :, j // 2] |= (values[:, :, j] & 15) << (4 * (j % 2))
                            repacked[:, :, 32 + j // 4] |= (values[:, :, j] >> 4) << (2 * (j % 4))
                        if not np.array_equal(packed, repacked):
                            raise ValueError('Q6 round-trip verification failed')
                        out.write(values.reshape(count // 4, 4, groups, 32, 2).transpose(0, 2, 3, 1, 4).tobytes())
                    for row in range(0, rows, 64):
                        count = min(rows - row, 64)
                        # Byte-level shuffle preserves every FP32 scale bit.
                        scales = np.frombuffer(read(count * groups * 4), np.uint8)
                        out.write(scales.reshape(count // 4, 4, groups, 4).transpose(0, 2, 1, 3).tobytes())

                matrix(vocab, d)
                for layer in range(nl):
                    copy(d * 8)
                    if header[80 + layer]:
                        copy(d * 3 * 4)
                        matrix(3 * d, d)
                        matrix(d, d)
                    else:
                        copy(hd * 8)
                        matrix(d, d)
                        matrix(nk * hd, d)
                        matrix(nk * hd, d)
                        matrix(d, d)
                    matrix(ff, d)
                    matrix(ff, d)
                    matrix(d, ff)
                copy(d * 4)
                if src.read(1):
                    raise ValueError('Unexpected trailing model data')
                if out.tell() >= 2**31:
                    raise ValueError('Expanded model exceeds the Win32 file limit')
        except BaseException:
            output.unlink()
            raise
    return {'source_sha256': sha256(source), 'output_sha256': sha256(output),
            'source_bytes': source.stat().st_size, 'bytes': output.stat().st_size,
            'format': 'SLMODEL1 v3 Q6X4; four interleaved rows; biased byte storage',
            'quantization_changed': False, 'all_packed_weight_bits_roundtrip_verified': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = expand(args.source, args.output)
    report = json.dumps(result, indent=2) + '\n'
    if args.report:
        args.report.write_text(report)
    print(report, end='')


if __name__ == '__main__':
    main()
