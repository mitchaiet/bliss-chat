#!/usr/bin/env python3
"""Build the bliss_pretrain_curated_v2 mixture from interchange JSONL conversations.

Input: a directory of *.jsonl files, one JSON object per line:
    {"messages":[{"role":"user","content":...},{"role":"assistant","content":...},...],
     "category": "...", "notes": ["...", ...]?}

Output: parquet shards with a single `text` column in the exact runtime formats
of NC_RUN.EXE v1.3.0, ready for BLISS_CURATED_PARQUET_DIR batch mixing:

  plain          "Q: u1\nA:a1\n\nQ: u2\nA:a2\n"
  system         SYSTEM + "\n" + plain                      (~30% of non-notes docs)
  notes          SYSTEM + "\nNotes: f1; f2\n" + plain       (all docs that carry notes)
  rollover       SYSTEM + "\nQ: Conversation so far: Q:q1; A:a1\n\nQ: u\nA:a\n"
                 (synthesized from ~5% of >=3-turn conversations)

Optionally copies the proven v1 curated shards into the output dir so the
mixture keeps the original single-turn behaviors.

Usage:
  python3 tools/build_bliss_pretrain_curated_v2.py \
      --src ~/bliss-chat-data/interchange_v2 \
      --out ~/bliss-chat-data/bliss_pretrain_curated_v2 \
      --v1-dir ~/bliss-chat-data/bliss_pretrain_curated_v1 \
      --val-jsonl ~/bliss-chat-data/bliss_v2_val.jsonl
"""
import argparse
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

SYSTEM = ("You are Bliss, a small local chat assistant on Windows XP. "
          "Answer in one short factual sentence.")

ASCII_RE = re.compile(r"^[\x20-\x7e]*$")


def normalize(text: str) -> str:
    # Best-effort ASCII projection, then whitespace cleanup.
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


def valid_conv(conv: dict, errors: list) -> bool:
    msgs = conv.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2 or len(msgs) % 2 != 0:
        errors.append("bad-shape")
        return False
    for i, m in enumerate(msgs):
        want = "user" if i % 2 == 0 else "assistant"
        if m.get("role") != want:
            errors.append("bad-alternation")
            return False
        c = m.get("content", "")
        if not isinstance(c, str) or not c.strip():
            errors.append("empty-turn")
            return False
    for i in range(1, len(msgs), 2):
        a = normalize(msgs[i]["content"])
        if len(a) > 220:
            errors.append("answer-too-long")
            return False
        if not a or a[-1] not in ".!?":
            errors.append("answer-no-terminal")
            return False
        words = a.lower().split()
        for n in (3,):
            grams = [" ".join(words[j:j + n]) for j in range(len(words) - n + 1)]
            if grams and max(grams.count(g) for g in set(grams)) >= 3:
                errors.append("answer-3gram-loop")
                return False
    for note in conv.get("notes", []) or []:
        if not isinstance(note, str) or not note.strip() or len(normalize(note)) > 120:
            errors.append("bad-note")
            return False
    return True


def render_plain(msgs) -> str:
    parts = []
    for i in range(0, len(msgs), 2):
        u = normalize(msgs[i]["content"])
        a = normalize(msgs[i + 1]["content"])
        parts.append(f"Q: {u}\nA:{a}")
    return "\n\n".join(parts) + "\n"


def render_doc(conv: dict, rng: random.Random) -> str:
    msgs = conv["messages"]
    notes = [normalize(n) for n in (conv.get("notes") or []) if normalize(n)]
    if notes:
        return f"{SYSTEM}\nNotes: {'; '.join(notes)}\n{render_plain(msgs)}"
    # rollover-summary synthesis for a slice of longer conversations
    if len(msgs) >= 6 and rng.random() < 0.05:
        cut = len(msgs) - 2
        summary = "; ".join(
            ("Q:" + normalize(msgs[i]["content"])[:100]) if i % 2 == 0
            else ("A:" + normalize(msgs[i]["content"])[:100])
            for i in range(cut)
        )
        u = normalize(msgs[cut]["content"])
        a = normalize(msgs[cut + 1]["content"])
        return (f"{SYSTEM}\nQ: Conversation so far: {summary}\n\n"
                f"Q: {u}\nA:{a}\n")
    if rng.random() < 0.30:
        return f"{SYSTEM}\n{render_plain(msgs)}"
    return render_plain(msgs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--v1-dir", default="")
    ap.add_argument("--val-jsonl", default="")
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=20260706)
    ap.add_argument("--rows-per-shard", type=int, default=25000)
    args = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = random.Random(args.seed)
    src = Path(args.src).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    convs, seen, stats, errors = [], set(), {}, []
    for path in sorted(src.glob("*.jsonl")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                conv = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"{path.name}:{line_no} json-error")
                continue
            if not valid_conv(conv, errors):
                continue
            key = "\x1f".join(normalize(m["content"]).lower() for m in conv["messages"])
            if key in seen:
                errors.append("dup")
                continue
            seen.add(key)
            convs.append(conv)
            cat = conv.get("category", "uncat")
            stats[cat] = stats.get(cat, 0) + 1

    rng.shuffle(convs)
    n_val = int(len(convs) * args.val_frac)
    val, train = convs[:n_val], convs[n_val:]

    docs = [render_doc(c, rng) for c in train]
    rng.shuffle(docs)

    shard_idx = 0
    for i in range(0, len(docs), args.rows_per_shard):
        table = pa.table({"text": docs[i:i + args.rows_per_shard]})
        pq.write_table(table, out / f"v2_shard_{shard_idx:05d}.parquet", row_group_size=2048)
        shard_idx += 1

    copied = 0
    if args.v1_dir:
        import shutil
        v1 = Path(args.v1_dir).expanduser()
        for p in sorted(v1.glob("*.parquet")):
            shutil.copy2(p, out / f"v1_{p.name}")
            copied += 1

    if args.val_jsonl:
        with open(Path(args.val_jsonl).expanduser(), "w", encoding="ascii") as f:
            for c in val:
                f.write(json.dumps(c, ensure_ascii=True) + "\n")

    manifest = {
        "format": "plain Q:/A: runtime text, v1.3.0 memory/context variants",
        "system_prefix": SYSTEM,
        "train_conversations": len(train),
        "val_conversations": len(val),
        "train_docs": len(docs),
        "v1_shards_copied": copied,
        "categories": stats,
        "rejects": len(errors),
        "seed": args.seed,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    from collections import Counter
    print("top rejects:", Counter(e.split()[-1] for e in errors).most_common(8), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
