#!/usr/bin/env python3
"""
Score bench/results.csv on three rubrics, each 0 or 1:

  coherent  — response has reasonable length and no severe repetition
  format    — response looks like a real answer (not empty, not just punctuation,
              not bleeding the "Q:" stop marker through)
  correct   — at least one expected_keyword appears in the response, case-insens.

Prints a per-category summary plus the totals. Writes bench/scored.csv with
per-question scores so we can diff against future runs.
"""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results.csv"
SCORED  = ROOT / "bench" / "scored.csv"


def _tokens(text: str) -> list[str]:
    return re.findall(r"\b[\w']+\b", text.lower())


def repetition_features(text: str) -> dict:
    """Return explicit repetition/ramble diagnostics for one response.

    The old coherence gate only caught repeated bigrams. Bliss' bad runs also
    collapse into single-token loops ("the the the") or longer phrase loops, so
    expose metrics that can be tracked in scored.csv and Linear notes.
    """
    words = _tokens(text)
    n = len(words)
    unique_ratio = (len(set(words)) / n) if n else 0.0

    max_token_run = 0
    cur_run = 0
    prev = None
    for w in words:
        if w == prev:
            cur_run += 1
        else:
            cur_run = 1
            prev = w
        if cur_run > max_token_run:
            max_token_run = cur_run

    max_ngram_repeats = 0
    repeated_ngram = ""
    for size in (2, 3, 4):
        if n < size * 2:
            continue
        counts = defaultdict(int)
        for i in range(n - size + 1):
            ng = tuple(words[i:i + size])
            counts[ng] += 1
            if counts[ng] > max_ngram_repeats:
                max_ngram_repeats = counts[ng]
                repeated_ngram = " ".join(ng)

    ramble_reason = ""
    if max_token_run >= 4:
        ramble_reason = "token_run"
    elif max_ngram_repeats >= 3:
        ramble_reason = "ngram_repeat"
    elif n >= 24 and unique_ratio < 0.35:
        ramble_reason = "low_unique_ratio"

    return {
        "token_count": n,
        "unique_token_ratio": f"{unique_ratio:.3f}" if n else "0.000",
        "max_token_run": max_token_run,
        "max_ngram_repeats": max_ngram_repeats,
        "repeated_ngram": repeated_ngram,
        "ramble": 1 if ramble_reason else 0,
        "ramble_reason": ramble_reason,
    }


def looks_repetitive(text: str) -> bool:
    return repetition_features(text)["ramble"] == 1


def score_row(row: dict) -> dict:
    resp = (row.get("response") or "").strip()
    low  = resp.lower()
    keywords_field = (row.get("expected_keywords") or "").lower()
    keywords = [k.strip() for k in keywords_field.split("|") if k.strip()]

    # coherent
    coherent = 1
    if len(resp) < 2:
        coherent = 0
    if len(resp) > 600:
        coherent = 0
    if looks_repetitive(resp):
        coherent = 0

    # format
    format_ok = 1
    if not resp:
        format_ok = 0
    if re.search(r"\bQ:\s*$", resp) or low.endswith("\nq:"):
        # Stop-pattern leaked through
        format_ok = 0
    if all(not c.isalnum() for c in resp):
        format_ok = 0

    # correct
    correct = 0
    for kw in keywords:
        if kw and kw in low:
            correct = 1
            break

    feats = repetition_features(resp)

    return {
        **row,
        "coherent": coherent,
        "format":   format_ok,
        "correct":  correct,
        "ramble": feats["ramble"],
        "ramble_reason": feats["ramble_reason"],
        "response_token_count": feats["token_count"],
        "unique_token_ratio": feats["unique_token_ratio"],
        "max_token_run": feats["max_token_run"],
        "max_ngram_repeats": feats["max_ngram_repeats"],
        "repeated_ngram": feats["repeated_ngram"],
    }


def main():
    if not RESULTS.exists():
        print(f"no {RESULTS}; run bench/run_bench.py first")
        sys.exit(1)
    rows = []
    with open(RESULTS) as f:
        for r in csv.DictReader(f):
            rows.append(score_row(r))

    # Per-category aggregation.
    cats = defaultdict(lambda: {
        "n": 0, "coherent": 0, "format": 0, "correct": 0, "ramble": 0,
    })
    for r in rows:
        c = cats[r["category"]]
        c["n"] += 1
        c["coherent"] += r["coherent"]
        c["format"]   += r["format"]
        c["correct"]  += r["correct"]
        c["ramble"]   += r["ramble"]

    print()
    print(f"{'category':<12} {'n':>3} {'coh%':>6} {'fmt%':>6} {'cor%':>6} {'rmb%':>6}")
    print("-" * 47)
    total = {"n": 0, "coherent": 0, "format": 0, "correct": 0, "ramble": 0}
    for cat, c in sorted(cats.items()):
        print(f"{cat:<12} {c['n']:>3} "
              f"{100 * c['coherent'] / c['n']:>5.1f}% "
              f"{100 * c['format']   / c['n']:>5.1f}% "
              f"{100 * c['correct']  / c['n']:>5.1f}% "
              f"{100 * c['ramble']   / c['n']:>5.1f}%")
        for k in total:
            total[k] += c[k]
    print("-" * 47)
    n = total["n"]
    print(f"{'TOTAL':<12} {n:>3} "
          f"{100 * total['coherent'] / n:>5.1f}% "
          f"{100 * total['format']   / n:>5.1f}% "
          f"{100 * total['correct']  / n:>5.1f}% "
          f"{100 * total['ramble']   / n:>5.1f}%")

    with open(SCORED, "w", newline="") as g:
        wtr = csv.DictWriter(g, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"\nwrote {SCORED}")


if __name__ == "__main__":
    main()
