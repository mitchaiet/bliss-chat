import re
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "nc_run.c"


class NcRunGuardrailSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SRC.read_text()

    def test_help_exposes_runtime_guardrail_controls(self):
        self.assertIn('/rambleguard <0|1>', self.source)
        self.assertIn('/repeat <f>', self.source)

    def test_runtime_defaults_are_safe_but_disableable(self):
        self.assertRegex(self.source, r"int\s+ramble_guard\s*=\s*1\s*;")
        self.assertRegex(self.source, r"float\s+repeat_penalty\s*=\s*1\.10f\s*;")

    def test_generation_stops_before_four_identical_tokens(self):
        self.assertIn('guard_should_stop', self.source)
        self.assertIn('max_token_run >= 4', self.source)
        self.assertIn('guardrail: repetition stopped', self.source)

    def test_sampler_can_apply_repetition_penalty_from_recent_tokens(self):
        self.assertIn('apply_repetition_penalty', self.source)
        self.assertRegex(self.source, r"sample\(&S,\s*temp,\s*top_p,\s*recent_ids")

    def test_thread_context_is_kept_between_turns_until_reset(self):
        # The backend must not restore the clean prefix before every normal
        # prompt; otherwise the model cannot use earlier messages in a thread.
        self.assertIn('Multi-turn: keep the KV cache across turns', self.source)
        self.assertNotIn('Fresh one-shot semantics for each user turn', self.source)
        self.assertNotRegex(self.source, r"state_restore_prefix\(&S\);\s*turn_idx\s*=\s*0;\s*char shaped_line")
        self.assertIn('context full, rebuilt from bounded thread summary', self.source)


if __name__ == "__main__":
    unittest.main()
