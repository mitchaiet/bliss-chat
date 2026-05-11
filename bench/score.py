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


def looks_repetitive(text: str) -> bool:
    """Catch the classic 'X is X is X' mode-collapse loop. Triggers if any
    contiguous bigram appears 3+ times in the response."""
    words = re.findall(r"\b[\w']+\b", text.lower())
    if len(words) < 6:
        return False
    bigrams = [tuple(words[i:i + 2]) for i in range(len(words) - 1)]
    counts = defaultdict(int)
    for bg in bigrams:
        counts[bg] += 1
        if counts[bg] >= 3:
            return True
    return False


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

    return {
        **row,
        "coherent": coherent,
        "format":   format_ok,
        "correct":  correct,
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
    cats = defaultdict(lambda: {"n": 0, "coherent": 0, "format": 0, "correct": 0})
    for r in rows:
        c = cats[r["category"]]
        c["n"] += 1
        c["coherent"] += r["coherent"]
        c["format"]   += r["format"]
        c["correct"]  += r["correct"]

    print()
    print(f"{'category':<12} {'n':>3} {'coh%':>6} {'fmt%':>6} {'cor%':>6}")
    print("-" * 40)
    total = {"n": 0, "coherent": 0, "format": 0, "correct": 0}
    for cat, c in sorted(cats.items()):
        print(f"{cat:<12} {c['n']:>3} "
              f"{100 * c['coherent'] / c['n']:>5.1f}% "
              f"{100 * c['format']   / c['n']:>5.1f}% "
              f"{100 * c['correct']  / c['n']:>5.1f}%")
        for k in total:
            total[k] += c[k]
    print("-" * 40)
    n = total["n"]
    print(f"{'TOTAL':<12} {n:>3} "
          f"{100 * total['coherent'] / n:>5.1f}% "
          f"{100 * total['format']   / n:>5.1f}% "
          f"{100 * total['correct']  / n:>5.1f}%")

    with open(SCORED, "w", newline="") as g:
        wtr = csv.DictWriter(g, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"\nwrote {SCORED}")


if __name__ == "__main__":
    main()
