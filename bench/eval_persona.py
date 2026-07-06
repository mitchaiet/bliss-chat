#!/usr/bin/env python3
"""
Persona probe runner for bench/bliss_native_eval_v1.jsonl.

Each JSONL line is {"prompt": <user line>, "expect_contains": <substring>}.
The backend is spawned ONCE; /reset is sent before every probe so each is
scored as a clean one-shot answer (this is the run_bench.py protocol, not
the multi-turn one -- see eval_multiturn_memory.py for that).

Scoring (case-insensitive):
    pass = expect_contains appears in the answer
    if expect_contains is empty ("Tell me a joke."), pass = any non-empty
    answer -- there is nothing meaningful to substring-match.

Run:  python3 bench/eval_persona.py
Prints one line per probe, failures, and "PERSONA SCORE: <passed>/<n>".
Writes bench/results_persona.csv.
"""
import argparse
import csv
import json
import select
import subprocess
import sys
import time
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
PROBES  = ROOT / "bench" / "bliss_native_eval_v1.jsonl"
RESULTS = ROOT / "bench" / "results_persona.csv"

READY_TIMEOUT = 300.0   # model load + few-shot prefill
TURN_TIMEOUT  = 180.0   # per-turn generation


# --------------------------------------------------------------------------
# Protocol plumbing (adapted from bench/run_bench.py).
# --------------------------------------------------------------------------
def strip_sentinels(raw):
    """Return visible text for one turn's worth of bytes."""
    parts = raw.split(b"\x01")
    out = []
    for p in parts:
        if not p:
            continue
        nl = p.find(b"\n")
        if nl < 0:
            out.append(p)
            continue
        sentinel = p[:nl]
        rest = p[nl + 1:]
        if sentinel.startswith((b"PROG ", b"INFO ", b"READY", b"EOT")):
            out.append(rest)
        else:
            out.append(p)
    return b"".join(out).decode("utf-8", errors="replace").strip()


class Backend:
    """One nc_run_native process speaking the \\x01-sentinel protocol."""

    def __init__(self, argv):
        self.argv = argv
        self.proc = None
        self.spawn()

    def spawn(self):
        self.proc = subprocess.Popen(
            self.argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self._read_until(lambda b: b"\x01READY" in b, READY_TIMEOUT, "READY")

    def respawn(self):
        self.close()
        self.spawn()

    def _read_until(self, done, timeout, what):
        buf = b""
        deadline = time.time() + timeout
        while not done(buf):
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(
                    "timed out after %.0fs waiting for %s; tail=%r"
                    % (timeout, what, buf[-120:]))
            r, _, _ = select.select([self.proc.stdout], [], [],
                                    min(remaining, 1.0))
            if not r:
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        "backend exited (code %s) waiting for %s"
                        % (self.proc.returncode, what))
                continue
            chunk = self.proc.stdout.read(4096)
            if not chunk:
                raise RuntimeError("backend closed stdout waiting for " + what)
            buf += chunk
        return buf

    def turn(self, prompt, timeout=TURN_TIMEOUT):
        """Send one line, return the stripped visible answer."""
        self.proc.stdin.write(prompt.encode("utf-8") + b"\n")
        self.proc.stdin.flush()
        raw = self._read_until(
            lambda b: b"\x01EOT" in b and b.endswith(b"\n"), timeout, "EOT")
        return strip_sentinels(raw)

    def close(self):
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        self.proc = None


def load_probes(path):
    probes = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            probes.append({"prompt": row["prompt"],
                           "expect": row.get("expect_contains", "")})
    return probes


def score_answer(answer, expect):
    if not expect:                       # nothing to match: any non-empty
        return bool(answer.strip())      # answer counts as a pass
    return expect.lower() in answer.lower()


def parse_args():
    ap = argparse.ArgumentParser(
        description="Persona probes from bliss_native_eval_v1.jsonl "
                    "(/reset between probes, substring-scored).")
    ap.add_argument("--model", default=str(ROOT / "build" / "deploy" / "MODEL.NCB"))
    ap.add_argument("--tokenizer", default=str(ROOT / "build" / "deploy" / "TOKENIZER.NCT"))
    ap.add_argument("--binary", default=str(ROOT / "build" / "nc_run_native"))
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    for p in (args.binary, args.model, args.tokenizer):
        if not Path(p).exists():
            print("missing artifact: %s (run scripts/build-xp.sh first)" % p)
            sys.exit(1)
    if not PROBES.exists():
        print("missing probe file: %s" % PROBES)
        sys.exit(1)

    probes = load_probes(PROBES)

    argv = [args.binary, args.model, args.tokenizer,
            "-c", str(args.ctx), "-t", str(args.temp), "-s", str(args.seed)]

    print("loading model + prefilling few-shot prefix...", flush=True)
    backend = Backend(argv)
    print("ready.", flush=True)

    rows = []
    failures = []
    passed_n = 0

    for i, probe in enumerate(probes):
        error = None
        try:
            backend.turn("/reset")       # clean one-shot answer per probe
            answer = backend.turn(probe["prompt"])
        except (TimeoutError, RuntimeError) as e:
            error = str(e)
            try:
                backend.respawn()
            except (TimeoutError, RuntimeError) as e2:
                print("fatal: could not respawn backend: %s" % e2)
                sys.exit(2)

        if error is not None:
            answer = "[backend error: %s]" % error
            passed = False
        else:
            passed = score_answer(answer, probe["expect"])

        if passed:
            passed_n += 1
        else:
            failures.append((i, probe["prompt"], probe["expect"], answer))

        rows.append({"probe_idx": i, "prompt": probe["prompt"],
                     "expect_contains": probe["expect"],
                     "passed": int(passed), "answer": answer})
        print("[%d/%d] %-4s %-32s %r" % (
            i + 1, len(probes), "PASS" if passed else "FAIL",
            probe["prompt"][:32], answer[:60]), flush=True)

    backend.close()

    with open(RESULTS, "w", newline="") as g:
        wtr = csv.DictWriter(g, fieldnames=["probe_idx", "prompt",
                                            "expect_contains", "passed",
                                            "answer"])
        wtr.writeheader()
        wtr.writerows(rows)

    if failures:
        print()
        for idx, q, exp, a in failures:
            print("FAIL probe %d: %r (expect %r) -> got: %r"
                  % (idx, q, exp, a[:120]))
    print("\nwrote %s" % RESULTS)
    print("PERSONA SCORE: %d/%d" % (passed_n, len(probes)))


if __name__ == "__main__":
    main()
