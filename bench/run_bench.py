#!/usr/bin/env python3
"""
Drive the current model through bench/questions.csv. For each question:
- spawn nc_run_native (so we don't pay model-load cost per question)
- restore the few-shot prefix between questions via /reset
- capture the assistant response
Result is written to bench/results.csv with question, response, latency.

Run:  python3 bench/run_bench.py
"""
import csv
import os
import re
import select
import subprocess
import sys
import time
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
BACKEND   = ROOT / "build" / "nc_run_native"
MODEL     = ROOT / "build" / "deploy" / "MODEL.NCB"
TOKENIZER = ROOT / "build" / "deploy" / "TOKENIZER.NCT"
QUESTIONS = ROOT / "bench" / "questions.csv"
RESULTS   = ROOT / "bench" / "results.csv"

# Use a low temperature so the same question is reproducible across runs.
TEMP = "0.3"
SEED = "42"


def strip_sentinels(raw: bytes) -> tuple[str, int]:
    """Return (visible_text, eot_token_count) for one turn's worth of bytes."""
    parts = raw.split(b"\x01")
    out = []
    tcount = 0
    for p in parts:
        if not p:
            continue
        nl = p.find(b"\n")
        if nl < 0:
            out.append(p)
            continue
        sentinel = p[:nl]
        rest = p[nl + 1 :]
        if sentinel.startswith((b"PROG ", b"INFO ", b"READY")):
            out.append(rest)
        elif sentinel.startswith(b"EOT"):
            try:
                tcount = int(sentinel.split()[-1])
            except Exception:
                pass
            out.append(rest)
        else:
            out.append(p)
    text = b"".join(out).decode("utf-8", errors="replace")
    return text.strip(), tcount


def main():
    if not BACKEND.exists() or not MODEL.exists() or not TOKENIZER.exists():
        print("missing artifacts; run scripts/build-xp.sh first")
        sys.exit(1)

    proc = subprocess.Popen(
        [str(BACKEND), str(MODEL), str(TOKENIZER),
         "-t", TEMP, "-s", SEED],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )

    # Wait for READY sentinel before sending anything.
    print("loading model + prefilling few-shot prefix...", flush=True)
    buf = b""
    while b"\x01READY" not in buf:
        chunk = proc.stdout.read(64)
        if not chunk:
            print("backend died during load")
            sys.exit(2)
        buf += chunk
    print("ready.")

    def turn(prompt: str) -> tuple[str, int, float]:
        """Send one prompt, return (visible_text, token_count, elapsed_s)."""
        proc.stdin.write(prompt.encode() + b"\n")
        proc.stdin.flush()
        t0 = time.time()
        out = b""
        while True:
            chunk = proc.stdout.read1(4096) if hasattr(proc.stdout, "read1") else proc.stdout.read(4096)
            if not chunk:
                break
            out += chunk
            if b"\x01EOT" in out and out.endswith(b"\n"):
                break
        elapsed = time.time() - t0
        text, tc = strip_sentinels(out)
        return text, tc, elapsed

    with open(QUESTIONS) as f, open(RESULTS, "w", newline="") as g:
        rdr = csv.DictReader(f)
        wtr = csv.DictWriter(g, fieldnames=[
            "id", "category", "question", "expected_keywords",
            "response", "tokens", "elapsed_s",
        ])
        wtr.writeheader()
        for i, row in enumerate(rdr, 1):
            # Reset the conversation between questions so we get a clean
            # one-shot answer to each. The /reset returns immediately.
            turn("/reset")
            resp, tc, dt = turn(row["question"])
            print(f"[{i:3d}/100] {row['category']:10s} {row['question'][:50]:50s} -> "
                  f"{resp[:60]!r} ({tc}t, {dt:.1f}s)", flush=True)
            wtr.writerow({
                **row,
                "response": resp,
                "tokens": tc,
                "elapsed_s": f"{dt:.2f}",
            })

    proc.stdin.close()
    proc.wait(timeout=5)
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
