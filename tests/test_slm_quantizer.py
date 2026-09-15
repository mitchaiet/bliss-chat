#!/usr/bin/env python3
"""Weight-error and representation checks for the optional export quantizer."""
import importlib.util
from pathlib import Path
import unittest
import torch

spec = importlib.util.spec_from_file_location("export_slm", Path(__file__).resolve().parents[1] / "tools/export_slm.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class QuantizerTests(unittest.TestCase):
    def test_error_and_packing(self):
        torch.manual_seed(20260915)
        for bits in (4, 8):
            w = torch.cat([torch.zeros(1, 64), torch.ones(1, 64),
                torch.randn(1024, 64), torch.randn(1024, 64) ** 3])
            w[2, 0] = 100
            old_q, old_s = exporter.quantize_groups(w, bits, "maxabs")
            q, s = exporter.quantize_groups(w, bits, "mse")
            reconstructed = q.reshape_as(w).float() * s[:, None]
            old = old_q.reshape_as(w).float() * old_s[:, None]
            error, old_error = (w - reconstructed).square().mean(1), (w - old).square().mean(1)
            self.assertTrue(torch.isfinite(reconstructed).all())
            self.assertTrue((s > 0).all())
            self.assertTrue((error <= old_error + 1e-7).all())
            self.assertLess(error.mean(), old_error.mean())
            if bits == 4:
                unsigned = (q.to(torch.int16) + 8).to(torch.uint8)
                packed = unsigned[::2] | (unsigned[1::2] << 4)
                unpacked = torch.stack([packed & 15, packed >> 4], dim=1).flatten().to(torch.int16) - 8
                self.assertTrue(torch.equal(unpacked, q))


if __name__ == "__main__":
    torch.set_num_threads(8)
    unittest.main()
