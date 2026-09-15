#!/usr/bin/env python3
"""Run CPU-native IPC, saved-note persistence, invalid-input recovery and replay checks."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--binary", required=True)
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--runtime-arg", action="append", default=[])
    p.add_argument("--limit-mib", type=int, help="Linux child virtual-memory limit, including mapped weights")
    a = p.parse_args()
    constrain = None
    if a.limit_mib is not None:
        import resource
        import sys
        if sys.platform != "linux" or a.limit_mib < 1:
            raise ValueError("Memory limit requires Linux and a positive MiB value")
        def constrain():
            limit = a.limit_mib * 1048576
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    checks = {}
    with tempfile.TemporaryDirectory() as directory:
        memory = Path(directory) / "MEMORY.TXT"
        args = [a.binary, str(a.export / "MODEL.SLM"), str(a.export / "TOKENIZER.SLT"), "-c", "512", "-t", "0", "-s", "42", "-n", "12", "-m", str(memory)] + a.runtime_arg
        def run(commands, extra=()):
            r = subprocess.run(args + list(extra), input="\n".join(commands) + "\n", text=True, capture_output=True, timeout=120, preexec_fn=constrain)
            if r.returncode:
                raise AssertionError({"exit": r.returncode, "stderr": r.stderr, "stdout": r.stdout})
            if "\x01READY\n" not in r.stdout:
                raise AssertionError("READY missing")
            return r.stdout
        text = run(["/remember My dog is named Pepper.", "/remember My favorite color is green.", "/memories"])
        checks["saved_notes"] = memory.read_text() == "My dog is named Pepper.\nMy favorite color is green.\n"
        checks["command_endings"] = text.count("\x01EOT 0") == 3
        text = run(["/memories", "/forget 2", "/memreload", "/memories"])
        checks["restart_reload_forget"] = "1. My dog is named Pepper." in text and memory.read_text() == "My dog is named Pepper.\n"
        text = run(["/system " + "oversized " * 600, "oversized " * 700, "/info", "/replay My cat is blue.\tYour cat is blue.", "/reset", "/info"])
        checks["oversized_system_recoverable"] = "previous prompt retained" in text
        checks["oversized_user_recoverable"] = "Prompt too long" in text and text.count("\x01EOT 0") == 6
        checks["replay_reset_alive"] = text.count("ChatML, Q") == 2
        text = run(["What is my dog's name?"])
        checks["persistent_note_used"] = "Pepper" in text
        checks["generated_turn_ended"] = "\x01EOT " in text
        samples = {"persistent_note_reply": text}
        text = run(["/replay My kite is yellow.\tYour kite is yellow."] * 8 + ["/info", "What color is my kite?"], ["-c", "128", "--system", "Answer briefly."])
        checks["context_rollover_alive"] = "yellow" in text.lower() and text.count("\x01EOT ") == 10
        samples["rollover_reply"] = text[-1000:]
    report = {"checks": checks, "samples": samples, "linux_virtual_memory_limit_mib": a.limit_mib, "runtime_args": a.runtime_arg}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(checks, indent=2))
    raise SystemExit(not all(checks.values()))


if __name__ == "__main__":
    main()
