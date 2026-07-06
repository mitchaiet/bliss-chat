#!/usr/bin/env python3
"""
Persistent-notes (-m memfile) eval for nc_run_native.

25 cases. Each case writes its notes to a temp file (one note per line,
the format mem_load() expects), spawns a FRESH backend with -m <tempfile>,
waits for READY, asks exactly one question, scores the answer, and
terminates the process. A fresh process per case guarantees the notes are
picked up via the prefilled "Notes: a; b; c" prefix line and that no
in-conversation state leaks between cases.

Two kinds of case:
    answered  (20)  the question is answered by exactly ONE of the notes.
                    "reject" holds distinctive words from the OTHER notes,
                    so an engine that dumps all its notes fails.
    unrelated (5)   the question has nothing to do with the notes.
                    "expect" is a general-knowledge keyword; "reject" holds
                    note words -- the notes must not bleed into the answer.

Scoring (case-insensitive):
    pass = (at least one "expect" substring present)
       AND (no "reject" substring present)

Run:  python3 bench/eval_notes.py
Prints one line per case, failures, and "NOTES SCORE: <passed>/25".
Writes bench/results_notes.csv.
"""
import argparse
import csv
import os
import select
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results_notes.csv"

READY_TIMEOUT = 300.0   # model load + few-shot prefill (paid per case)
TURN_TIMEOUT  = 180.0   # per-turn generation

# --------------------------------------------------------------------------
# Note sets. Kept <=12 notes and <=159 chars per line (mem_load limits).
# --------------------------------------------------------------------------
SET_A = ["My name is Harold.", "My dog is called Waffles.",
         "I work as an electrician."]
SET_B = ["My favorite color is orange.", "My sister lives in Miami.",
         "I drive a green jeep."]
SET_C = ["My birthday is in March.", "My cat is named Pepper.",
         "My favorite food is tacos."]
SET_D = ["I live in Austin.", "My brother is named Miguel.",
         "My favorite sport is hockey."]
SET_E = ["My parrot is named Kiwi.", "I work as a nurse.",
         "My car is yellow."]
SET_F = ["My daughter is named Rosa.", "My favorite drink is cocoa.",
         "I live on Maple Street."]
SET_G = ["My hamster is called Biscotti.", "My favorite season is winter.",
         "I play the trumpet."]

CASES = [
    # ---------------- answered-by-one-note (20) ----------------
    {"kind": "answered", "notes": SET_A,
     "question": "What is my name?",
     "expect": ["Harold"], "reject": ["Waffles", "electric"]},
    {"kind": "answered", "notes": SET_A,
     "question": "What is my dog's name?",
     "expect": ["Waffles"], "reject": ["Harold", "electric"]},
    {"kind": "answered", "notes": SET_A,
     "question": "What is my job?",
     "expect": ["electric"], "reject": ["Harold", "Waffles"]},

    {"kind": "answered", "notes": SET_B,
     "question": "What is my favorite color?",
     "expect": ["orange"], "reject": ["Miami", "jeep"]},
    {"kind": "answered", "notes": SET_B,
     "question": "Where does my sister live?",
     "expect": ["Miami"], "reject": ["orange", "jeep"]},
    {"kind": "answered", "notes": SET_B,
     "question": "What do I drive?",
     "expect": ["jeep"], "reject": ["orange", "Miami"]},

    {"kind": "answered", "notes": SET_C,
     "question": "Which month is my birthday in?",
     "expect": ["March"], "reject": ["Pepper", "taco"]},
    {"kind": "answered", "notes": SET_C,
     "question": "What is my cat called?",
     "expect": ["Pepper"], "reject": ["March", "taco"]},
    {"kind": "answered", "notes": SET_C,
     "question": "What is my favorite food?",
     "expect": ["taco"], "reject": ["March", "Pepper"]},

    {"kind": "answered", "notes": SET_D,
     "question": "Which city do I live in?",
     "expect": ["Austin"], "reject": ["Miguel", "hockey"]},
    {"kind": "answered", "notes": SET_D,
     "question": "What is my brother's name?",
     "expect": ["Miguel"], "reject": ["Austin", "hockey"]},
    {"kind": "answered", "notes": SET_D,
     "question": "What is my favorite sport?",
     "expect": ["hockey"], "reject": ["Austin", "Miguel"]},

    {"kind": "answered", "notes": SET_E,
     "question": "What is my parrot's name?",
     "expect": ["Kiwi"], "reject": ["nurse", "yellow"]},
    {"kind": "answered", "notes": SET_E,
     "question": "What is my job?",
     "expect": ["nurse"], "reject": ["Kiwi", "yellow"]},
    {"kind": "answered", "notes": SET_E,
     "question": "What color is my car?",
     "expect": ["yellow"], "reject": ["Kiwi", "nurse"]},

    {"kind": "answered", "notes": SET_F,
     "question": "What is my daughter's name?",
     "expect": ["Rosa"], "reject": ["cocoa", "Maple"]},
    {"kind": "answered", "notes": SET_F,
     "question": "What is my favorite drink?",
     "expect": ["cocoa"], "reject": ["Rosa", "Maple"]},
    {"kind": "answered", "notes": SET_F,
     "question": "What street do I live on?",
     "expect": ["Maple"], "reject": ["Rosa", "cocoa"]},

    {"kind": "answered", "notes": SET_G,
     "question": "What is my hamster's name?",
     "expect": ["Biscotti"], "reject": ["winter", "trumpet"]},
    {"kind": "answered", "notes": SET_G,
     "question": "What instrument do I play?",
     "expect": ["trumpet"], "reject": ["Biscotti", "winter"]},

    # ---------------- unrelated-to-notes (5) ----------------
    {"kind": "unrelated", "notes": SET_A,
     "question": "What is the capital of France?",
     "expect": ["Paris"], "reject": ["Harold", "Waffles", "electric"]},
    {"kind": "unrelated", "notes": SET_C,
     "question": "What color is the sky?",
     "expect": ["blue"], "reject": ["March", "Pepper", "taco"]},
    {"kind": "unrelated", "notes": SET_D,
     "question": "How many legs does a spider have?",
     "expect": ["eight", "8"], "reject": ["Austin", "Miguel", "hockey"]},
    {"kind": "unrelated", "notes": SET_E,
     "question": "What do bees make?",
     "expect": ["honey"], "reject": ["Kiwi", "nurse", "yellow"]},
    {"kind": "unrelated", "notes": SET_F,
     "question": "What planet do we live on?",
     "expect": ["Earth"], "reject": ["Rosa", "cocoa", "Maple"]},
]


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
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self._read_until(lambda b: b"\x01READY" in b, READY_TIMEOUT, "READY")

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


# --------------------------------------------------------------------------
# Scoring.
# --------------------------------------------------------------------------
def score_answer(answer, expect=(), reject=()):
    low = answer.lower()
    ok_any = any(e.lower() in low for e in expect) if expect else True
    ok_rej = not any(r.lower() in low for r in reject)
    return ok_any and ok_rej


def parse_args():
    ap = argparse.ArgumentParser(
        description="Persistent-notes eval: fresh backend per case with "
                    "-m <notes file>, one question, substring-scored.")
    ap.add_argument("--model", default=str(ROOT / "build" / "deploy" / "MODEL.NCB"))
    ap.add_argument("--tokenizer", default=str(ROOT / "build" / "deploy" / "TOKENIZER.NCT"))
    ap.add_argument("--binary", default=str(ROOT / "build" / "nc_run_native"))
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def run_case(args, case):
    """Write notes file, spawn backend with -m, ask, tear down.
    Returns (answer_text, error_or_None)."""
    fd, notes_path = tempfile.mkstemp(prefix="bliss_notes_", suffix=".txt")
    backend = None
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for note in case["notes"]:
                f.write(note + "\n")
        argv = [args.binary, args.model, args.tokenizer,
                "-c", str(args.ctx), "-t", str(args.temp),
                "-s", str(args.seed), "-m", notes_path]
        backend = Backend(argv)
        answer = backend.turn(case["question"])
        return answer, None
    except (TimeoutError, RuntimeError) as e:
        return "", str(e)
    finally:
        if backend is not None:
            backend.close()
        try:
            os.unlink(notes_path)
        except OSError:
            pass


def main():
    args = parse_args()
    for p in (args.binary, args.model, args.tokenizer):
        if not Path(p).exists():
            print("missing artifact: %s (run scripts/build-xp.sh first)" % p)
            sys.exit(1)

    rows = []
    failures = []
    passed_n = 0

    for i, case in enumerate(CASES):
        answer, error = run_case(args, case)
        if error is not None:
            answer = "[backend error: %s]" % error
            passed = False
        else:
            passed = score_answer(answer, case["expect"], case.get("reject", ()))

        if passed:
            passed_n += 1
        else:
            failures.append((i, case["question"], answer))

        rows.append({"case_idx": i, "kind": case["kind"],
                     "question": case["question"],
                     "passed": int(passed), "answer": answer})
        print("[%2d/%d] %-9s %-4s %-38s %r" % (
            i + 1, len(CASES), case["kind"], "PASS" if passed else "FAIL",
            case["question"][:38], answer[:60]), flush=True)

    with open(RESULTS, "w", newline="") as g:
        wtr = csv.DictWriter(g, fieldnames=["case_idx", "kind", "question",
                                            "passed", "answer"])
        wtr.writeheader()
        wtr.writerows(rows)

    if failures:
        print()
        for idx, q, a in failures:
            print("FAIL case %d: %r -> got: %r" % (idx, q, a[:120]))
    print("\nwrote %s" % RESULTS)
    print("NOTES SCORE: %d/%d" % (passed_n, len(CASES)))


if __name__ == "__main__":
    main()
