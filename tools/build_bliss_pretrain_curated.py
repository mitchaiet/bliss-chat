#!/usr/bin/env python3
"""Build Bliss-native plain-text pretraining documents and eval prompts.

The intent is to mix these documents into nanochat base pretraining from step 1,
using the exact runtime-friendly format we want the XP model to learn:

    Q: <user question>
    A:<short Bliss answer>

No chat-special tokens, no markdown tables, no long assistant essays.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
from pathlib import Path


def load_v5_rows(repo: Path) -> list[list[dict[str, str]]]:
    path = repo / "tools" / "build_bliss_sft_v5_data.py"
    spec = importlib.util.spec_from_file_location("build_bliss_sft_v5_data", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod.build_rows()


def clean(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    return s.encode("ascii", "ignore").decode("ascii")


def to_doc(conv: list[dict[str, str]]) -> str:
    user = clean(conv[0]["content"])
    assistant = clean(conv[1]["content"])
    # Runtime uses no-space A: prefix.
    return f"Q: {user}\nA:{assistant}\n"


def make_docs(rows: list[list[dict[str, str]]], count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    base = [to_doc(r) for r in rows]
    docs: list[str] = []
    # Preserve the curated rows, then sample with replacement for the requested count.
    while len(docs) < count:
        d = rng.choice(base)
        # Tiny harmless prompt-format augmentation; keep answer untouched.
        if rng.random() < 0.18:
            d = d.replace("Q: ", "Q: Answer briefly: ", 1)
        elif rng.random() < 0.12:
            d = d.replace("\nA:", "\nA:", 1)
        docs.append(d)
    rng.shuffle(docs)
    return docs


def write_jsonl(path: Path, docs: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, text in enumerate(docs):
            f.write(json.dumps({"id": f"bliss-pretrain-{i:06d}", "text": text}, ensure_ascii=True) + "\n")


def write_parquet(path: Path, docs: list[str]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"text": docs})
    pq.write_table(table, path, row_group_size=1024, compression="zstd")


def write_eval(path: Path) -> None:
    prompts = [
        {"prompt": "What is your name?", "expect_contains": "I am Bliss"},
        {"prompt": "Who are you?", "expect_contains": "Bliss"},
        {"prompt": "Where are you running?", "expect_contains": "Windows XP"},
        {"prompt": "What is a computer?", "expect_contains": "machine"},
        {"prompt": "Can you browse the web?", "expect_contains": "No"},
        {"prompt": "What is 2 plus 2?", "expect_contains": "4"},
        {"prompt": "Say the word cat five times.", "expect_contains": "cat"},
        {"prompt": "Tell me a joke.", "expect_contains": ""},
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in prompts:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out-dir", type=Path, default=Path("data/bliss_pretrain_curated_v1"))
    ap.add_argument("--count", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=20260512)
    ap.add_argument("--parquet", action="store_true")
    args = ap.parse_args()

    repo = args.repo.resolve()
    out = args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir
    rows = load_v5_rows(repo)
    docs = make_docs(rows, args.count, args.seed)
    write_jsonl(out / "train.jsonl", docs)
    write_eval(repo / "bench" / "bliss_native_eval_v1.jsonl")
    manifest = {
        "name": "bliss_pretrain_curated_v1",
        "source": "tools/build_bliss_sft_v5_data.py",
        "format": "plain Q:/A: runtime text",
        "count": len(docs),
        "seed": args.seed,
        "recommended_mix_probability": 0.12,
        "notes": "Mix into base pretraining from step 1; avoid custom chat special tokens for this experiment.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.parquet:
        write_parquet(out / "shard_00000.parquet", docs)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
