#!/usr/bin/env python3
"""
Export a nanochat checkpoint (.pt + meta.json) to NCB1 binary format
suitable for the XP inference engine.

Layout (little-endian):

  Header (256 bytes, padded with zeros):
    char[8]  magic            "NCB1\0\0\0\0"
    int32    version          1
    int32    dtype             0=fp32, 1=int8 (per-row scale)
    int32    vocab_size
    int32    pad_vocab_size
    int32    n_layer
    int32    n_head
    int32    n_kv_head
    int32    n_embd
    int32    head_dim
    int32    sequence_len
    int32    rotary_base       int(_) of the rotary base (e.g. 100000)
    int32    rotary_seq_len
    uint64   ve_layer_mask     bit i set => layer i has a value embedding
    uint64   window_pattern_mask  bit i set => layer i is "L" (full); else "S"
    int32    short_window
    int32    long_window
    int32    smear_gate_in     24
    int32    ve_gate_in        12
    int32    softcap           15
    (rest zero-padded to 256)

  Weights, all little-endian, contiguous, in this exact order:
    wte                    (pad_vocab_size, n_embd)            embed
    For layer i in 0..n_layer-1:
      attn.c_q.weight        (n_head*head_dim, n_embd)
      attn.c_k.weight        (n_kv_head*head_dim, n_embd)
      attn.c_v.weight        (n_kv_head*head_dim, n_embd)
      attn.c_proj.weight     (n_embd, n_embd)
      mlp.c_fc.weight        (4*n_embd, n_embd)
      mlp.c_proj.weight      (n_embd, 4*n_embd)
      if has_ve(i):
        attn.ve_gate.weight  (n_kv_head, 12)
    For each VE layer i (in increasing order):
      value_embeds[i].weight (pad_vocab_size, n_kv_head*head_dim)
    resid_lambdas          (n_layer,)            fp32
    x0_lambdas             (n_layer,)            fp32
    smear_gate.weight      (1, 24)               fp32 / weight dtype
    smear_lambda           (1,)                  fp32
    backout_lambda         (1,)                  fp32
    lm_head.weight         (pad_vocab_size, n_embd)
    cos                    (rotary_seq_len, head_dim/2)  fp32
    sin                    (rotary_seq_len, head_dim/2)  fp32

Usage:
    python export_ncb.py --src /path/to/d6 --step 1500 --out /path/to/MODEL.NCB
    python export_ncb.py --src /path/to/d6 --step 1500 --out /path/to/MODEL.NCB --int8
"""
import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch

MAGIC = b"NCB1\x00\x00\x00\x00"
VERSION = 1
HEADER_SIZE = 256


def has_ve(layer_idx: int, n_layer: int) -> bool:
    return layer_idx % 2 == (n_layer - 1) % 2


def write_header(f, *, dtype_code: int, cfg: dict, head_dim: int,
                 ve_mask: int, win_mask: int,
                 short_window: int, long_window: int,
                 rotary_base: int, rotary_seq_len: int):
    pad_vocab = cfg["pad_vocab_size"]
    parts = [MAGIC,
             struct.pack("<i", VERSION),
             struct.pack("<i", dtype_code),
             struct.pack("<i", cfg["vocab_size"]),
             struct.pack("<i", pad_vocab),
             struct.pack("<i", cfg["n_layer"]),
             struct.pack("<i", cfg["n_head"]),
             struct.pack("<i", cfg["n_kv_head"]),
             struct.pack("<i", cfg["n_embd"]),
             struct.pack("<i", head_dim),
             struct.pack("<i", cfg["sequence_len"]),
             struct.pack("<i", rotary_base),
             struct.pack("<i", rotary_seq_len),
             struct.pack("<Q", ve_mask),
             struct.pack("<Q", win_mask),
             struct.pack("<i", short_window),
             struct.pack("<i", long_window),
             struct.pack("<i", 24),  # smear_gate_in
             struct.pack("<i", 12),  # ve_gate_in
             struct.pack("<i", 15),  # softcap
             ]
    blob = b"".join(parts)
    assert len(blob) <= HEADER_SIZE
    f.write(blob)
    f.write(b"\x00" * (HEADER_SIZE - len(blob)))


def write_tensor_fp32(f, t: torch.Tensor):
    arr = t.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy()
    f.write(arr.tobytes(order="C"))


def quantize_int8_per_row(t: torch.Tensor):
    """
    Per-row symmetric int8 quantization. Returns (qweight int8, scale fp32).
    Treats t as 2D: (rows, cols). 1D tensors are quantized as a single row.
    """
    arr = t.detach().to(dtype=torch.float32, device="cpu").contiguous()
    if arr.ndim == 1:
        arr = arr.unsqueeze(0)
    abs_max = arr.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = (abs_max / 127.0)  # (rows, 1)
    q = (arr / scale).round().clamp(-127, 127).to(torch.int8)
    return q.numpy(), scale.squeeze(-1).numpy()  # (rows, cols), (rows,)


def write_tensor_int8(f, t: torch.Tensor):
    q, s = quantize_int8_per_row(t)
    # Layout: [int8 weights row-major][fp32 scales]
    f.write(q.tobytes(order="C"))
    f.write(s.astype(np.float32).tobytes(order="C"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="checkpoint dir, e.g. /path/to/nanochat-cache/chat_checkpoints/d6")
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: highest)")
    ap.add_argument("--out", required=True, help="output .ncb path")
    ap.add_argument("--int8", action="store_true", help="quantize matrix weights to int8 per-row")
    args = ap.parse_args()

    src = Path(os.path.expanduser(args.src))
    if not src.exists():
        sys.exit(f"src dir not found: {src}")

    # Find checkpoint
    if args.step is None:
        steps = sorted(int(p.stem.split("_")[-1]) for p in src.glob("model_*.pt"))
        if not steps:
            sys.exit(f"no model_*.pt in {src}")
        step = steps[-1]
    else:
        step = args.step
    pt_path = src / f"model_{step:06d}.pt"
    meta_path = src / f"meta_{step:06d}.json"
    if not pt_path.exists():
        sys.exit(f"missing {pt_path}")
    if not meta_path.exists():
        sys.exit(f"missing {meta_path}")

    meta = json.loads(meta_path.read_text())
    cfg_kwargs = meta["model_config"]
    print(f"[export] loading {pt_path}", file=sys.stderr)
    print(f"[export] config: {cfg_kwargs}", file=sys.stderr)

    sd = torch.load(pt_path, map_location="cpu", weights_only=True)

    # Resolve config + derived sizes
    n_layer = int(cfg_kwargs["n_layer"])
    n_head = int(cfg_kwargs["n_head"])
    n_kv_head = int(cfg_kwargs["n_kv_head"])
    n_embd = int(cfg_kwargs["n_embd"])
    vocab_size = int(cfg_kwargs["vocab_size"])
    sequence_len = int(cfg_kwargs["sequence_len"])
    window_pattern = cfg_kwargs.get("window_pattern", "L")
    head_dim = n_embd // n_head
    kv_dim = n_kv_head * head_dim

    # Padded vocab size from the actual embedding tensor
    pad_vocab = sd["transformer.wte.weight"].shape[0]
    cfg = {
        "vocab_size": vocab_size,
        "pad_vocab_size": pad_vocab,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_kv_head": n_kv_head,
        "n_embd": n_embd,
        "sequence_len": sequence_len,
    }

    # VE mask
    ve_mask = 0
    for i in range(n_layer):
        if has_ve(i, n_layer):
            ve_mask |= (1 << i)

    # Window pattern mask (1 = L = long, 0 = S = short).
    # Mirror nanochat: tile the pattern, then force last layer to L.
    pat = (window_pattern.upper() or "L")
    win_mask = 0
    for i in range(n_layer):
        c = pat[i % len(pat)]
        if c == "L":
            win_mask |= (1 << i)
    win_mask |= (1 << (n_layer - 1))  # last layer always L
    long_window = sequence_len
    # ceil(seq_len / 4 / 128) * 128, mirror nanochat
    short_window = -(-long_window // 4 // 128) * 128

    # Rotary precompute
    rotary_seq_len = sequence_len * 10
    rotary_base = 100000  # nanochat default

    # ----- write file -----
    out = Path(os.path.expanduser(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    dtype_code = 1 if args.int8 else 0
    write = write_tensor_int8 if args.int8 else write_tensor_fp32

    print(f"[export] writing {out} (dtype={'int8' if args.int8 else 'fp32'}, pad_vocab={pad_vocab})", file=sys.stderr)

    with out.open("wb") as f:
        write_header(f,
                     dtype_code=dtype_code,
                     cfg=cfg,
                     head_dim=head_dim,
                     ve_mask=ve_mask,
                     win_mask=win_mask,
                     short_window=short_window,
                     long_window=long_window,
                     rotary_base=rotary_base,
                     rotary_seq_len=rotary_seq_len)

        # Token embeddings: quantize when --int8 (per-row scale)
        write(f, sd["transformer.wte.weight"])

        # Layers
        for i in range(n_layer):
            base = f"transformer.h.{i}"
            write(f, sd[f"{base}.attn.c_q.weight"])
            write(f, sd[f"{base}.attn.c_k.weight"])
            write(f, sd[f"{base}.attn.c_v.weight"])
            write(f, sd[f"{base}.attn.c_proj.weight"])
            write(f, sd[f"{base}.mlp.c_fc.weight"])
            write(f, sd[f"{base}.mlp.c_proj.weight"])
            if has_ve(i, n_layer):
                # ve_gate is tiny (n_kv_head x 12) — keep fp32 for accuracy & simplicity
                write_tensor_fp32(f, sd[f"{base}.attn.ve_gate.weight"])

        # Value embeddings: quantize when --int8 (per-row scale, lookup is cheap)
        for i in range(n_layer):
            if has_ve(i, n_layer):
                write(f, sd[f"value_embeds.{i}.weight"])

        # Per-layer residual scalars (fp32)
        write_tensor_fp32(f, sd["resid_lambdas"])
        write_tensor_fp32(f, sd["x0_lambdas"])

        # Smear gate weight + scalars (fp32 — tiny)
        write_tensor_fp32(f, sd["smear_gate.weight"])
        write_tensor_fp32(f, sd["smear_lambda"])
        write_tensor_fp32(f, sd["backout_lambda"])

        # LM head (matrix, quantized if requested)
        write(f, sd["lm_head.weight"])

        # Precomputed RoPE cos/sin — match nanochat's _precompute_rotary_embeddings.
        # channel_range = arange(0, head_dim, 2)
        # inv_freq      = 1 / (base ** (channel_range / head_dim))
        # t = arange(rotary_seq_len)
        # freqs = outer(t, inv_freq)  ; cos = cos(freqs), sin = sin(freqs)
        ch = torch.arange(0, head_dim, 2, dtype=torch.float64)
        inv_freq = 1.0 / (rotary_base ** (ch / head_dim))
        t = torch.arange(rotary_seq_len, dtype=torch.float64)
        freqs = torch.outer(t, inv_freq)  # (rotary_seq_len, head_dim/2)
        cos = freqs.cos().to(torch.float32).contiguous()
        sin = freqs.sin().to(torch.float32).contiguous()
        f.write(cos.numpy().tobytes(order="C"))
        f.write(sin.numpy().tobytes(order="C"))

    size = out.stat().st_size
    print(f"[export] wrote {size} bytes ({size/1e6:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
