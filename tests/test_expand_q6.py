"""Round-trip every tensor in a tiny hybrid model; reject unsafe conversions."""
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('expand_q6', Path(__file__).resolve().parents[1] / 'tools/expand_q6.py')
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def fixture():
    header = bytearray(256)
    struct.pack_into('<8s14IffII', header, 0, b'SLMODEL1', 2, 6, 64, 128, 2, 2, 1, 32, 256, 512, 64, 1, 7, 1, 1000000., 1e-5, 3, 1)
    header[80:82] = bytes([1, 0])
    pieces, records = [header], []

    def copy(size):
        data = bytes((i * 71 + len(records)) % 256 for i in range(size))
        records.append(('copy', data))
        pieces.append(data)

    def matrix(rows, cols):
        values = np.fromfunction(lambda r, c: (r * 13 + c * 7 + len(records)) % 64, (rows, cols), dtype=int).astype(np.uint8)
        groups = values.reshape(rows, cols // 64, 64)
        packed = bytearray()
        for row in groups:
            for group in row:
                packed.extend((int(group[j]) & 15) | ((int(group[j + 1]) & 15) << 4) for j in range(0, 64, 2))
                packed.extend(sum((int(group[j + k]) >> 4) << (2 * k) for k in range(4)) for j in range(0, 64, 4))
        scales = (np.arange(rows * cols // 64, dtype=np.float32).reshape(rows, cols // 64) + 1) / 1000
        pieces.extend([packed, scales.tobytes()])
        records.append(('matrix', values, scales))

    matrix(256, 64)
    for layer in range(2):
        copy(64 * 8)
        if layer == 0:
            copy(64 * 3 * 4)
            matrix(192, 64)
            matrix(64, 64)
        else:
            copy(32 * 8)
            for rows in (64, 32, 32, 64):
                matrix(rows, 64)
        matrix(128, 64)
        matrix(128, 64)
        matrix(64, 128)
    copy(64 * 4)
    return b''.join(pieces), records


class ConversionTest(unittest.TestCase):
    def test_every_value_scale_and_nonmatrix_byte(self):
        source, records = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / 'input', Path(tmp) / 'output'
            src.write_bytes(source)
            result = converter.expand(src, out)
            self.assertTrue(result['all_packed_weight_bits_roundtrip_verified'])
            self.assertEqual(src.read_bytes(), source)
            data = out.read_bytes()
        self.assertEqual(struct.unpack_from('<II', data, 8), (3, 8))
        self.assertEqual(data[16:256], source[16:256])
        offset = 256
        for record in records:
            if record[0] == 'copy':
                expected = record[1]
                self.assertEqual(data[offset:offset + len(expected)], expected)
                offset += len(expected)
                continue
            _, values, scales = record
            rows, cols = values.shape
            # Read via the published address formula, independent of converter transpose.
            for r in range(rows):
                for c in range(cols):
                    address = offset + ((r // 4) * (cols // 64) + c // 64) * 256 + (c % 64 // 2) * 8 + (r % 4) * 2 + c % 2
                    self.assertEqual(data[address], int(values[r, c]))
            offset += rows * cols
            for r in range(rows):
                for g in range(cols // 64):
                    address = offset + ((r // 4) * (cols // 64) + g) * 16 + (r % 4) * 4
                    self.assertEqual(data[address:address + 4], scales[r, g].tobytes())
            offset += scales.nbytes
        self.assertEqual(offset, len(data))

    def test_rejects_corruption_and_preserves_existing_output(self):
        source, _ = fixture()
        invalid = [source[:80], source[:-1], source + b'trailing']
        for offset, value in [(8, 3), (12, 8), (48, 32), (24, 129), (36, 3), (40, 257)]:
            bad = bytearray(source)
            struct.pack_into('<I', bad, offset, value)
            invalid.append(bad)
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / 'input', Path(tmp) / 'output'
            for bad in invalid:
                src.write_bytes(bad)
                with self.assertRaises(ValueError):
                    converter.expand(src, out)
                self.assertFalse(out.exists())
            src.write_bytes(source)
            out.write_bytes(b'keep me')
            with self.assertRaises(FileExistsError):
                converter.expand(src, out)
            self.assertEqual(out.read_bytes(), b'keep me')


if __name__ == '__main__':
    unittest.main()
