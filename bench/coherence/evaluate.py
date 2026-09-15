#!/usr/bin/env python3
"""Sealed, model-independent Bliss cases; POSIX sentinel runner and offline scoring.

No GPU use. Fresh process per case; only temporary, fictional notes are used.
All output directories must be new. Do not put this directory in training globs.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import selectors
import statistics
import subprocess
import tempfile
import time

CASES = Path(__file__).with_name("heldout_v1.jsonl")
SENTINEL = re.compile(rb"\x01[^\n]*\n")
EOT = re.compile(rb"\x01EOT(?: ([^\n]*))?\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_cases(path):
    cases = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    assert len({c["id"] for c in cases}) == len(cases), "duplicate case IDs"
    for c in cases:
        assert c["turns"] and all(isinstance(t, str) and t and "\n" not in t for t in c["turns"])
        assert c.get("checks") and c.get("reference") and c.get("rubric")
    return cases


def score(answer, checks):
    """Mechanical content/format gate; never a substitute for semantic review."""
    answer = answer.strip()
    flags = 0 if checks.get("case_sensitive") else re.IGNORECASE
    failures = []
    if not answer:
        failures.append("empty")
    for pattern in checks.get("all", []):
        if not re.search(pattern, answer, flags):
            failures.append("required:" + pattern)
    if checks.get("any") and not any(re.search(p, answer, flags) for p in checks["any"]):
        failures.append("no accepted alternative")
    for pattern in checks.get("none", []):
        if re.search(pattern, answer, flags):
            failures.append("forbidden:" + pattern)
    if "exact" in checks:
        observed = answer if checks.get("case_sensitive") else answer.casefold()
        expected = checks["exact"] if checks.get("case_sensitive") else checks["exact"].casefold()
        if observed != expected:
            failures.append("exact output mismatch")
    if "full" in checks and not re.fullmatch(checks["full"], answer, flags):
        failures.append("full output mismatch")
    if "max_words" in checks and len(answer.split()) > checks["max_words"]:
        failures.append("too many words")
    if any(s in answer for s in ("\x01", "<|im_start|>", "<|im_end|>", "\nQ:", "\nA:")):
        failures.append("protocol or role leakage")
    return {"objective_pass": not failures, "objective_failures": failures}


def visible(raw):
    # Drop an incomplete sentinel as well as complete sentinel lines, so progress
    # messages split across pipe reads cannot count as the first visible token.
    tail = raw.rfind(b"\x01")
    if tail >= 0 and b"\n" not in raw[tail:]:
        raw = raw[:tail]
    return SENTINEL.sub(b"", raw).decode("utf-8", "replace").strip()


class Backend:
    def __init__(self, argv, stderr_file, timeout):
        self.proc = None
        self.sel = selectors.DefaultSelector()
        self.peak_rss_kib = 0
        self.stderr_file = stderr_file
        self.timeout = timeout
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, bufsize=0)
        self.sel.register(self.proc.stdout, selectors.EVENT_READ, "out")
        self.sel.register(self.proc.stderr, selectors.EVENT_READ, "err")
        started = time.perf_counter()
        try:
            self.startup = self.read_until("ready", started)
        except Exception:
            self.close()
            raise

    def memory(self):
        try:
            status = Path(f"/proc/{self.proc.pid}/status").read_text()
            value = re.search(r"^VmHWM:\s+(\d+)", status, re.M)
            if value:
                self.peak_rss_kib = max(self.peak_rss_kib, int(value[1]))
        except (OSError, ValueError):
            pass

    def read_until(self, target, started):
        raw = b""
        first_visible_s = None
        last_visible_s = None
        old_visible = ""
        while True:
            elapsed = time.perf_counter() - started
            if elapsed > self.timeout:
                raise TimeoutError(f"Timed out waiting for {target}; visible tail: {visible(raw)[-100:]!r}")
            self.memory()
            for key, _ in self.sel.select(min(0.05, self.timeout - elapsed)):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    self.sel.unregister(key.fileobj)
                    continue
                if key.data == "err":
                    self.stderr_file.write(chunk)
                    self.stderr_file.flush()
                    continue
                raw += chunk
                text = visible(raw)
                if text and text != old_visible:
                    stamp = time.perf_counter() - started
                    if first_visible_s is None:
                        first_visible_s = stamp
                    last_visible_s = stamp
                    old_visible = text
            done = (b"\x01READY\n" in raw) if target == "ready" else EOT.search(raw)
            if done:
                self.memory()
                count = 0
                if target != "ready":
                    match = EOT.search(raw)
                    try:
                        count = int((match[1] or b"0").split()[-1])
                    except ValueError:
                        pass
                return {"answer": visible(raw), "tokens": count,
                        "elapsed_s": time.perf_counter() - started,
                        "ttft_visible_s": first_visible_s,
                        "last_visible_s": last_visible_s,
                        "sentinels": [x.decode("utf-8", "replace").strip() for x in SENTINEL.findall(raw)]}
            if not self.sel.get_map():
                raise RuntimeError(f"Backend exited before {target}; code={self.proc.poll()}")

    def turn(self, prompt):
        started = time.perf_counter()
        self.proc.stdin.write(prompt.encode("utf-8") + b"\n")
        self.proc.stdin.flush()
        return self.read_until("eot", started)

    def close(self):
        if self.proc is None:
            return
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=2)
        self.sel.close()
        self.proc.stdout.close()
        self.proc.stderr.close()
        self.proc = None


def summarize(rows):
    counts = collections.defaultdict(lambda: [0, 0])
    for row in rows:
        counts[row["category"]][0] += bool(row.get("objective_pass"))
        counts[row["category"]][1] += 1
    timing = {}
    for key in ("elapsed_s", "ttft_visible_s"):
        values = [t[key] for r in rows for t in r.get("turn_results", []) if t.get(key) is not None]
        if values:
            values.sort()
            timing[key] = {"median": statistics.median(values), "p95": values[math.ceil(len(values) * .95)-1]}
    return {"objective_pass": sum(bool(r.get("objective_pass")) for r in rows),
            "cases": len(rows), "by_category": dict(counts),
            "protocol_errors": sum(bool(r.get("error")) for r in rows),
            "peak_backend_rss_kib": max((r.get("peak_backend_rss_kib", 0) for r in rows), default=0),
            "all_user_turn_latency": timing,
            "semantic_review": "required; objective score alone does not establish coherence"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=Path, default=CASES)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--predictions", type=Path, help="Score external JSONL rows with id/answer; no inference")
    ap.add_argument("--binary", type=Path)
    ap.add_argument("--runtime-arg", action="append", default=[],
                    help="Append one native argument; repeat as needed (e.g. --runtime-arg=--float-activations)")
    ap.add_argument("--model", type=Path)
    ap.add_argument("--tokenizer", type=Path)
    ap.add_argument("--out", type=Path, help="New directory, never overwrites an old result")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--no-controls", action="store_true", help="Skip legacy /maxtok, /info and /template commands")
    ap.add_argument("--categories", help="Comma-separated diagnostic subset; full suite required for final comparison")
    args = ap.parse_args()
    cases = load_cases(args.cases)
    if args.validate:
        errors = {c["id"]: score(c["reference"], c["checks"]) for c in cases
                  if not score(c["reference"], c["checks"])["objective_pass"]}
        assert not errors, errors
        assert not score("No, the ticket is not valid.", {"all": [r"\bvalid\b"], "none": [r"\bnot valid\b"]})["objective_pass"]
        assert visible(b"\x01PROG 1\nhello\x01EO") == "hello"
        print(json.dumps({"validated_cases": len(cases), "categories": dict(collections.Counter(c["category"] for c in cases)),
                          "sha256": sha256(args.cases)}, indent=2))
        return
    if args.categories:
        selected = set(args.categories.split(","))
        cases = [c for c in cases if c["category"] in selected]
        if not cases:
            ap.error("No categories matched")
    if not args.out:
        ap.error("--out is required")
    if args.predictions:
        source = [json.loads(l) for l in args.predictions.read_text().splitlines() if l.strip()]
        assert len({r["id"] for r in source}) == len(source), "duplicate prediction IDs"
        by_id = {r["id"]: r for r in source}
        rows = []
        for case in cases:
            pred = by_id.get(case["id"], {})
            row = {**pred, "id": case["id"], "category": case["category"],
                   **score(pred.get("answer", ""), case["checks"])}
            if case["id"] not in by_id:
                row["error"] = "missing prediction"
            rows.append(row)
        args.out.mkdir(parents=True, exist_ok=False)
        metadata = {"cases_sha256": sha256(args.cases), "predictions_sha256": sha256(args.predictions),
                    "mode": "offline scoring", "extra_prediction_ids": sorted(set(by_id) - {c["id"] for c in cases})}
        (args.out / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    else:
        if not all((args.binary, args.model, args.tokenizer)):
            ap.error("--binary, --model and --tokenizer are required for inference")
        if os.name != "posix":
            ap.error("This pipe-timing runner supports POSIX; it does not validate Windows XP behavior")
        paths = {k: str(getattr(args, k).resolve(strict=True)) for k in ("binary", "model", "tokenizer")}
        args.out.mkdir(parents=True, exist_ok=False)
        metadata = {"mode": "POSIX CPU sentinel backend", "platform": platform.platform(),
                    "cases_sha256": sha256(args.cases), "files": {k: {"path": v, "sha256": sha256(v),
                    "bytes": Path(v).stat().st_size} for k, v in paths.items()},
                    "ctx": args.ctx, "temperature": 0, "seed": 42, "max_tokens": args.max_tokens,
                    "runtime_args": args.runtime_arg,
                    "controls_applied": not args.no_controls, "categories": args.categories or "all",
                    "timing_note": "TTFT measures first visible pipe bytes; EOT tokens/elapsed includes prefill, not pure decode speed"}
        rows = []
        with (args.out / "predictions.jsonl").open("x") as predictions:
            for case in cases:
                row = {"id": case["id"], "category": case["category"], "turn_results": []}
                backend = None
                with tempfile.TemporaryDirectory(prefix="bliss-eval-notes-") as tempdir, (args.out / (case["id"] + ".stderr.log")).open("xb") as stderr:
                    notes = Path(tempdir) / "notes.txt"
                    notes.write_text("\n".join(case.get("notes", [])) + "\n")
                    argv = [paths["binary"], paths["model"], paths["tokenizer"], "-c", str(args.ctx), "-t", "0", "-s", "42", "-m", str(notes)]
                    argv.extend(args.runtime_arg)
                    try:
                        backend = Backend(argv, stderr, args.timeout)
                        row["startup"] = backend.startup
                        if not args.no_controls:
                            row["controls"] = [backend.turn(command) for command in (f"/maxtok {args.max_tokens}", "/info", "/template")]
                        for prompt in case["turns"]:
                            row["turn_results"].append(backend.turn(prompt))
                        row["answer"] = row["turn_results"][-1]["answer"]
                    except (OSError, RuntimeError, TimeoutError) as error:
                        row["error"] = f"{type(error).__name__}: {error}"
                        row["answer"] = ""
                    finally:
                        if backend:
                            row["peak_backend_rss_kib"] = backend.peak_rss_kib
                            backend.close()
                    row.update(score(row["answer"], case["checks"]))
                    rows.append(row)
                    predictions.write(json.dumps(row) + "\n")
                    predictions.flush()
                    print(f"{case['id']}: {'PASS' if row['objective_pass'] else 'FAIL'} {row['answer'][:120]!r}", flush=True)
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summary = summarize(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Gold/rubrics live only in the separate review file, not prediction rows.
    review = []
    indexed = {r["id"]: r for r in rows}
    for case in cases:
        review.append({"id": case["id"], "category": case["category"], "notes": case.get("notes", []),
                       "turns": case["turns"], "answer": indexed[case["id"]].get("answer", ""),
                       "reference": case["reference"], "rubric": case["rubric"],
                       "relevance_0_2": None, "correctness_0_2": None, "coherence_0_2": None,
                       "instruction_0_2": None, "review_note": ""})
    (args.out / "review.jsonl").write_text("".join(json.dumps(r) + "\n" for r in review))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
