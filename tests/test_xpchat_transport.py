"""Compile the current GUI's pure C transport helpers and run portable checks.

Run: python3 tests/test_xpchat_transport.py -v
Requires a host C compiler (clang by default); does not require MinGW or Windows.
XPCHAT_TEST_CC selects a compiler; XPCHAT_TEST_SANITIZERS=0 disables ASan/UBSan.
The harnesses exercise actual source excerpts, not copied implementations.
"""

import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def excerpt(source, begin, end):
    if source.count(begin) != 1 or source.count(end) != 1:
        raise ValueError(f"GUI source marker missing or ambiguous: {begin!r}, {end!r}")
    start = source.index(begin)
    finish = source.index(end, start)
    return source[start:finish]


class XpChatTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "src" / "xpchat.c").read_text(encoding="utf-8")
        print("xpchat.c SHA256", hashlib.sha256(cls.source.encode()).hexdigest(), flush=True)

    def compile_and_run(self, name, replacements, expected):
        template = (ROOT / "tests" / f"{name}.c.in").read_text(encoding="utf-8")
        for marker, body in replacements.items():
            self.assertEqual(template.count(marker), 1, marker)
            template = template.replace(marker, body)
        with tempfile.TemporaryDirectory(prefix="bliss-gui-check-") as temporary:
            path = Path(temporary)
            source = path / f"{name}.c"
            binary = path / name
            source.write_text(template, encoding="utf-8")
            command = shlex.split(os.environ.get("XPCHAT_TEST_CC", "clang"))
            command += ["-std=c99", "-Wall", "-Wextra", "-Werror", "-g"]
            if os.environ.get("XPCHAT_TEST_SANITIZERS", "1") != "0":
                command += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
            command += [str(source), "-o", str(binary)]
            for operation in (command, [str(binary)]):
                result = subprocess.run(operation, text=True, capture_output=True, timeout=60)
                self.assertEqual(result.returncode, 0,
                                 f"{shlex.join(operation)}\n{result.stdout}\n{result.stderr}")
            self.assertEqual(result.stdout.strip(), expected)
            print(result.stdout.strip(), flush=True)

    def test_utf8_conversion_and_every_byte_split(self):
        conversion = excerpt(self.source, "static int utf8_unit(",
                             "static char *window_text_utf8(")
        # Windows WCHAR is 16 bits; macOS/Linux wchar_t need not be. Substitute
        # only its length primitive so the unchanged conversion uses UTF-16.
        self.assertEqual(conversion.count("wcslen(text)"), 1)
        conversion = conversion.replace("wcslen(text)", "u16len(text)")
        self.compile_and_run("xpchat_utf8_harness", {
            "/* GUI_CONVERSION_SOURCE */": conversion,
            "/* GUI_BUFFER_SOURCE */": excerpt(self.source, "static void buffer_init(",
                                                "static void normalize_newlines("),
            "/* GUI_STREAM_SOURCE */": excerpt(self.source, "static void post_utf8_chunk(",
                                                "// ---------- backend reader ----------"),
        }, "UTF8_GUI_BOUNDARY_CHECKS 79 passed")

    def test_control_acknowledgment_fifo(self):
        self.compile_and_run("xpchat_queue_harness", {
            "/* GUI_QUEUE_SOURCE */": excerpt(self.source, "#define REQUEST_QUEUE_SIZE",
                                               "static DWORD gRunStarted;"),
        }, "GUI_CONTROL_EOT_QUEUE 519 checks passed")


if __name__ == "__main__":
    unittest.main()
