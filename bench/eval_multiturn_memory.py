#!/usr/bin/env python3
"""
Multi-turn in-conversation memory eval for nc_run_native.

40 scripted dialogs across four categories:
    personal_fact (15)  fact stated early, distractor turns, then recall question
    pronoun_carry (10)  "Tell me about X." then a follow-up using it/they
    running_list  (8)   "add milk", "add eggs", "what is on my list?"
    correction    (7)   fact stated, then corrected; recall must use the new value

IMPORTANT: unlike run_bench.py, this eval does NOT /reset between the turns
of a dialog -- in-context memory is exactly what is under test. /reset is
sent exactly once at the start of each dialog.

Scoring (only the recall turn's answer is scored, all case-insensitive):
    pass = (at least one "expect" substring present, if "expect" given)
       AND (every "expect_all" substring present, if given -- running lists)
       AND (no "reject" substring present, if given -- corrections)

Run:  python3 bench/eval_multiturn_memory.py
Prints per-category pass counts, one line per failure, and a final
"MEMORY SCORE: <passed>/<total>". Writes bench/results_memory.csv.
"""
import argparse
import csv
import select
import subprocess
import sys
import time
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results_memory.csv"

READY_TIMEOUT = 300.0   # model load + few-shot prefill
TURN_TIMEOUT  = 180.0   # per-turn generation

# --------------------------------------------------------------------------
# Dialogs. "recall_turn" indexes into "turns"; only that turn's answer is
# scored. "expect" = any-of, "expect_all" = all-of, "reject" = none-of.
# --------------------------------------------------------------------------
DIALOGS = [
    # ---------------- personal_fact (15) ----------------
    {"category": "personal_fact",
     "turns": ["My name is Marcus.",
               "What is the capital of Italy?",
               "What is my name?"],
     "recall_turn": 2, "expect": ["Marcus"]},
    {"category": "personal_fact",
     "turns": ["I live in Denver.",
               "What color is the sky?",
               "Which city do I live in?"],
     "recall_turn": 2, "expect": ["Denver"]},
    {"category": "personal_fact",
     "turns": ["My dog is named Biscuit.",
               "What is the capital of France?",
               "How many legs does a spider have?",
               "What is my dog's name?"],
     "recall_turn": 3, "expect": ["Biscuit"]},
    {"category": "personal_fact",
     "turns": ["My favorite food is pizza.",
               "Is the sun hot?",
               "What is my favorite food?"],
     "recall_turn": 2, "expect": ["pizza"]},
    {"category": "personal_fact",
     "turns": ["My cat is called Whiskers.",
               "What is two plus two?",
               "What is my cat's name?"],
     "recall_turn": 2, "expect": ["Whiskers"]},
    {"category": "personal_fact",
     "turns": ["Hello there.",
               "My sister's name is Elena.",
               "What color is grass?",
               "What is my sister's name?"],
     "recall_turn": 3, "expect": ["Elena"]},
    {"category": "personal_fact",
     "turns": ["I am 34 years old.",
               "What is the capital of Japan?",
               "Name a fruit.",
               "How old am I?"],
     "recall_turn": 3, "expect": ["34", "thirty-four", "thirty four"]},
    {"category": "personal_fact",
     "turns": ["I drive a red truck.",
               "Do fish swim?",
               "What color is my truck?"],
     "recall_turn": 2, "expect": ["red"]},
    {"category": "personal_fact",
     "turns": ["My favorite color is purple.",
               "What is the largest ocean?",
               "How many days are in a week?",
               "What is my favorite color?"],
     "recall_turn": 3, "expect": ["purple"]},
    {"category": "personal_fact",
     "turns": ["My birthday is in October.",
               "What sound does a cow make?",
               "Which month is my birthday in?"],
     "recall_turn": 2, "expect": ["October"]},
    {"category": "personal_fact",
     "turns": ["Good morning.",
               "I work as a teacher.",
               "What is the capital of Spain?",
               "What is my job?"],
     "recall_turn": 3, "expect": ["teach"]},
    {"category": "personal_fact",
     "turns": ["My brother is called Felix.",
               "Is ice cold?",
               "What do bees make?",
               "What is my brother's name?"],
     "recall_turn": 3, "expect": ["Felix"]},
    {"category": "personal_fact",
     "turns": ["My hometown is Chicago.",
               "What planet do we live on?",
               "What is my hometown?"],
     "recall_turn": 2, "expect": ["Chicago"]},
    {"category": "personal_fact",
     "turns": ["My parrot is named Mango.",
               "How many legs does a dog have?",
               "What is the capital of Egypt?",
               "What is my parrot's name?"],
     "recall_turn": 3, "expect": ["Mango"]},
    {"category": "personal_fact",
     "turns": ["My favorite sport is tennis.",
               "What color are bananas?",
               "What is my favorite sport?"],
     "recall_turn": 2, "expect": ["tennis"]},

    # ---------------- pronoun_carry (10) ----------------
    {"category": "pronoun_carry",
     "turns": ["Tell me about Saturn.", "How many moons does it have?"],
     "recall_turn": 1, "expect": ["moon"]},
    {"category": "pronoun_carry",
     "turns": ["Tell me about dolphins.", "Where do they live?"],
     "recall_turn": 1, "expect": ["ocean", "sea", "water"]},
    {"category": "pronoun_carry",
     "turns": ["Tell me about penguins.", "Can they fly?"],
     "recall_turn": 1, "expect": ["fly"]},
    {"category": "pronoun_carry",
     "turns": ["Tell me about fire trucks.", "What color are they?"],
     "recall_turn": 1, "expect": ["red"]},
    {"category": "pronoun_carry",
     "turns": ["Tell me about elephants.", "What do they eat?"],
     "recall_turn": 1, "expect": ["plant", "grass", "leaf", "leaves", "vegetation"]},
    {"category": "pronoun_carry",
     "turns": ["Tell me about Paris.", "What country is it in?"],
     "recall_turn": 1, "expect": ["France"]},
    {"category": "pronoun_carry",
     "turns": ["Tell me about zebras.", "What color are their stripes?"],
     "recall_turn": 1, "expect": ["black", "white"]},
    {"category": "pronoun_carry",
     "turns": ["Tell me about the moon.", "What does it orbit?"],
     "recall_turn": 1, "expect": ["Earth"]},
    {"category": "pronoun_carry",
     "turns": ["Tell me about bees.", "What do they make?"],
     "recall_turn": 1, "expect": ["honey"]},
    {"category": "pronoun_carry",
     "turns": ["Tell me about spiders.", "How many legs do they have?"],
     "recall_turn": 1, "expect": ["eight", "8"]},

    # ---------------- running_list (8) ----------------
    {"category": "running_list",
     "turns": ["Add milk to my shopping list.",
               "Also add eggs to the list.",
               "What is on my list?"],
     "recall_turn": 2, "expect_all": ["milk", "egg"]},
    {"category": "running_list",
     "turns": ["Please add bread to my list.",
               "Add cheese to my list too.",
               "What is on my list?"],
     "recall_turn": 2, "expect_all": ["bread", "cheese"]},
    {"category": "running_list",
     "turns": ["Add apples to my list.",
               "Add bananas to my list.",
               "Add rice to my list.",
               "What is on my list?"],
     "recall_turn": 3, "expect_all": ["apple", "banana", "rice"]},
    {"category": "running_list",
     "turns": ["Add soap to my shopping list.",
               "Add towels as well.",
               "What is on my shopping list?"],
     "recall_turn": 2, "expect_all": ["soap", "towel"]},
    {"category": "running_list",
     "turns": ["Add coffee to my list.",
               "Also add sugar.",
               "What is on my list?"],
     "recall_turn": 2, "expect_all": ["coffee", "sugar"]},
    {"category": "running_list",
     "turns": ["Add a hammer to my list.",
               "Add nails to my list.",
               "What is on my list?"],
     "recall_turn": 2, "expect_all": ["hammer", "nail"]},
    {"category": "running_list",
     "turns": ["Add carrots to my list.",
               "Add onions too.",
               "What items are on my list?"],
     "recall_turn": 2, "expect_all": ["carrot", "onion"]},
    {"category": "running_list",
     "turns": ["Add batteries to my list.",
               "Add tape to my list.",
               "Add glue to my list.",
               "What is on my list?"],
     "recall_turn": 3, "expect_all": ["batter", "tape", "glue"]},

    # ---------------- correction (7) ----------------
    {"category": "correction",
     "turns": ["My cat is named Tom.",
               "Actually, her name is Luna.",
               "What is my cat called?"],
     "recall_turn": 2, "expect": ["Luna"], "reject": ["Tom"]},
    {"category": "correction",
     "turns": ["I live in Boston.",
               "Sorry, I meant to say I live in Seattle.",
               "Which city do I live in?"],
     "recall_turn": 2, "expect": ["Seattle"], "reject": ["Boston"]},
    {"category": "correction",
     "turns": ["My favorite color is green.",
               "Actually, my favorite color is blue.",
               "What is my favorite color?"],
     "recall_turn": 2, "expect": ["blue"], "reject": ["green"]},
    {"category": "correction",
     "turns": ["My dog is called Rex.",
               "Correction: his name is actually Bruno.",
               "What is my dog's name?"],
     "recall_turn": 2, "expect": ["Bruno"], "reject": ["Rex"]},
    {"category": "correction",
     "turns": ["My birthday is in June.",
               "Wait, that is wrong. My birthday is in April.",
               "Which month is my birthday in?"],
     "recall_turn": 2, "expect": ["April"], "reject": ["June"]},
    {"category": "correction",
     "turns": ["I work as a plumber.",
               "Actually I changed jobs, so now I am a baker.",
               "What is my job?"],
     "recall_turn": 2, "expect": ["baker"], "reject": ["plumber"]},
    {"category": "correction",
     "turns": ["My car is silver.",
               "Actually the car is green.",
               "What color is my car?"],
     "recall_turn": 2, "expect": ["green"], "reject": ["silver"]},
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


# --------------------------------------------------------------------------
# Scoring.
# --------------------------------------------------------------------------
def score_answer(answer, expect=(), expect_all=(), reject=()):
    low = answer.lower()
    ok_any = any(e.lower() in low for e in expect) if expect else True
    ok_all = all(e.lower() in low for e in expect_all)
    ok_rej = not any(r.lower() in low for r in reject)
    return ok_any and ok_all and ok_rej


def parse_args():
    ap = argparse.ArgumentParser(
        description="Multi-turn in-conversation memory eval (no /reset "
                    "between turns of a dialog).")
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

    argv = [args.binary, args.model, args.tokenizer,
            "-c", str(args.ctx), "-t", str(args.temp), "-s", str(args.seed)]

    print("loading model + prefilling few-shot prefix...", flush=True)
    backend = Backend(argv)
    print("ready.", flush=True)

    rows = []
    failures = []
    tally = {}  # category -> [passed, total]

    for i, d in enumerate(DIALOGS):
        answers = []
        error = None
        try:
            backend.turn("/reset")          # once per dialog, never mid-dialog
            for t in d["turns"]:
                answers.append(backend.turn(t))
        except (TimeoutError, RuntimeError) as e:
            error = str(e)
            try:
                backend.respawn()
            except (TimeoutError, RuntimeError) as e2:
                print("fatal: could not respawn backend: %s" % e2)
                sys.exit(2)

        question = d["turns"][d["recall_turn"]]
        if error is not None:
            answer = "[backend error: %s]" % error
            passed = False
        else:
            answer = answers[d["recall_turn"]]
            passed = score_answer(answer,
                                  d.get("expect", ()),
                                  d.get("expect_all", ()),
                                  d.get("reject", ()))

        cat = d["category"]
        tally.setdefault(cat, [0, 0])
        tally[cat][1] += 1
        if passed:
            tally[cat][0] += 1
        else:
            failures.append((i, question, answer))

        rows.append({"dialog_idx": i, "category": cat,
                     "question": question,
                     "passed": int(passed), "answer": answer})
        print("[%2d/%d] %-14s %-4s %r" % (
            i + 1, len(DIALOGS), cat, "PASS" if passed else "FAIL",
            answer[:70]), flush=True)

    backend.close()

    with open(RESULTS, "w", newline="") as g:
        wtr = csv.DictWriter(g, fieldnames=["dialog_idx", "category",
                                            "question", "passed", "answer"])
        wtr.writeheader()
        wtr.writerows(rows)

    print()
    total_pass = 0
    for cat in ("personal_fact", "pronoun_carry", "running_list", "correction"):
        p, n = tally.get(cat, (0, 0))
        total_pass += p
        print("%-14s %d/%d" % (cat, p, n))
    if failures:
        print()
        for idx, q, a in failures:
            print("FAIL dialog %d: %r -> got: %r" % (idx, q, a[:120]))
    print("\nwrote %s" % RESULTS)
    print("MEMORY SCORE: %d/%d" % (total_pass, len(DIALOGS)))


if __name__ == "__main__":
    main()
