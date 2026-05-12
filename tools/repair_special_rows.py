#!/usr/bin/env python3
"""Create a Bliss SFT-safe checkpoint by repairing chat-special rows.

The old repair copied <|bos|> into every chat-special row, which made the
roles indistinguishable. This script keeps <|bos|> intact and gives each other
special token a distinct direction with the same norm scale as trained rows.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import shutil
from pathlib import Path

import torch


DEFAULT_SPECIALS = [
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]


def latest_step(src: Path) -> int:
    steps = sorted(int(p.stem.split("_")[-1]) for p in src.glob("model_*.pt"))
    if not steps:
        raise SystemExit(f"no model_*.pt found in {src}")
    return steps[-1]


def load_specials(tokenizer_path: Path) -> dict[str, int]:
    with tokenizer_path.open("rb") as f:
        enc = pickle.load(f)
    specials = getattr(enc, "_special_tokens", None)
    if not specials:
        raise SystemExit(f"could not read _special_tokens from {tokenizer_path}")
    return dict(specials)


def orthogonal_rows(count: int, dim: int, bos_unit: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    mat = torch.randn(dim, count, generator=gen, dtype=torch.float32)
    mat = mat - bos_unit[:, None] * (bos_unit @ mat)[None, :]
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return q.T.contiguous()


def repair_tensor(
    tensor: torch.Tensor,
    ids: list[int],
    bos_id: int,
    seed: int,
    bos_blend: float,
) -> tuple[torch.Tensor, list[str]]:
    dtype = tensor.dtype
    work = tensor.float().clone()
    dim = work.shape[1]
    trained = work[:bos_id]
    median_norm = torch.linalg.vector_norm(trained, dim=1).median()
    bos = work[bos_id]
    bos_norm = torch.linalg.vector_norm(bos).clamp_min(1e-8)
    bos_unit = bos / bos_norm
    dirs = orthogonal_rows(len(ids), dim, bos_unit, seed)
    ortho_weight = math.sqrt(max(0.0, 1.0 - bos_blend * bos_blend))

    lines = []
    for row_i, tok_id in enumerate(ids):
        row = bos_blend * bos_unit + ortho_weight * dirs[row_i]
        row = row / torch.linalg.vector_norm(row).clamp_min(1e-8) * median_norm
        work[tok_id] = row
        cos_to_bos = torch.dot(row / torch.linalg.vector_norm(row), bos_unit).item()
        lines.append(f"id={tok_id} norm={torch.linalg.vector_norm(row).item():.4f} cos_bos={cos_to_bos:.4f}")
    return work.to(dtype=dtype), lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source checkpoint dir")
    ap.add_argument("--out", required=True, help="output checkpoint dir")
    ap.add_argument("--tokenizer", default="~/.cache/nanochat/tokenizer/tokenizer.pkl")
    ap.add_argument("--step", type=int, default=None, help="checkpoint step, default highest")
    ap.add_argument("--seed", type=int, default=20260511)
    ap.add_argument("--bos-blend", type=float, default=0.25, help="0=random orthogonal, 1=bos copy")
    args = ap.parse_args()

    src = Path(os.path.expanduser(args.src))
    out = Path(os.path.expanduser(args.out))
    tokenizer = Path(os.path.expanduser(args.tokenizer))
    step = latest_step(src) if args.step is None else args.step
    model_path = src / f"model_{step:06d}.pt"
    meta_path = src / f"meta_{step:06d}.json"
    if not model_path.exists() or not meta_path.exists():
        raise SystemExit(f"missing checkpoint files for step {step} in {src}")

    specials = load_specials(tokenizer)
    bos_id = specials["<|bos|>"]
    ids = [specials[name] for name in DEFAULT_SPECIALS if name in specials]
    if not ids:
        raise SystemExit("no target special ids resolved")

    print(f"[repair] loading {model_path}")
    sd = torch.load(model_path, map_location="cpu", weights_only=True)

    report: list[str] = []
    keys = ["transformer.wte.weight", "lm_head.weight"]
    keys.extend(k for k in sorted(sd) if k.startswith("value_embeds.") and k.endswith(".weight"))
    for key_i, key in enumerate(keys):
        fixed, lines = repair_tensor(sd[key], ids, bos_id, args.seed + key_i * 1009, args.bos_blend)
        sd[key] = fixed
        report.append(f"[{key}]")
        report.extend(lines)

    meta = json.loads(meta_path.read_text())
    meta.setdefault("repair", {})
    meta["repair"].update(
        {
            "tool": "tools/repair_special_rows.py",
            "source": str(src),
            "source_step": step,
            "seed": args.seed,
            "bos_blend": args.bos_blend,
            "repaired_special_ids": ids,
            "repaired_special_names": [name for name in DEFAULT_SPECIALS if name in specials],
        }
    )

    out.mkdir(parents=True, exist_ok=True)
    torch.save(sd, out / f"model_{step:06d}.pt")
    (out / f"meta_{step:06d}.json").write_text(json.dumps(meta, indent=2) + "\n")
    for opt in src.glob(f"optim_{step:06d}_rank*.pt"):
        shutil.copy2(opt, out / opt.name)
    (out / "special_repair_report.txt").write_text("\n".join(report) + "\n")
    print(f"[repair] wrote {out / f'model_{step:06d}.pt'}")
    print(f"[repair] report {out / 'special_repair_report.txt'}")


if __name__ == "__main__":
    main()
