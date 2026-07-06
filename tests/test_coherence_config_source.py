import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NC_RUN = ROOT / "src" / "nc_run.c"
XPCHAT = ROOT / "src" / "xpchat.c"


class CoherenceConfigSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nc = NC_RUN.read_text()
        cls.xp = XPCHAT.read_text()

    def test_backend_has_versioned_prompt_template_metadata(self):
        self.assertIn("BLISS_PROMPT_TEMPLATE_VERSION", self.nc)
        self.assertIn('"bliss-qa-short-v1"', self.nc)
        self.assertIn("prompt template = %s v%d", self.nc)
        self.assertIn("/template      = show prompt-template metadata", self.nc)

    def test_backend_has_model_defaults_and_defaults_command(self):
        self.assertIn("BLISS_MODEL_DEFAULTS_VERSION", self.nc)
        self.assertIn("default_model_config_for", self.nc)
        self.assertIn("model_config_source", self.nc)
        self.assertIn("/defaults      = re-apply model defaults", self.nc)
        self.assertRegex(self.nc, r"max_tokens\s*=\s*cfg\.max_tokens")

    def test_backend_has_named_presets_command(self):
        self.assertIn("bliss_presets[]", self.nc)
        for name in ("deterministic", "balanced", "creative"):
            self.assertIn(f'"{name}"', self.nc)
        self.assertIn("apply_preset_by_name", self.nc)
        self.assertIn("/preset <name> = apply deterministic|balanced|creative", self.nc)

    def test_xpchat_presets_are_named_and_apply_full_sampling_tuples(self):
        self.assertIn("apply_settings_preset", self.xp)
        self.assertIn("Preset deterministic", self.xp)
        self.assertIn("Preset balanced", self.xp)
        self.assertIn("Preset creative", self.xp)
        self.assertIn("SetDlgItemTextA(dlg, IDC_TOPP_EDIT", self.xp)
        self.assertIn("SetDlgItemTextA(dlg, IDC_MAXTOK_EDIT", self.xp)

    def test_xpchat_has_local_tools_before_model_generation(self):
        self.assertIn("try_answer_local_tool", self.xp)
        self.assertIn("append_local_assistant_answer", self.xp)
        self.assertIn('"Tool result:', self.xp)
        self.assertIn("safe_eval_arithmetic", self.xp)
        self.assertIn("GetLocalTime", self.xp)
        self.assertIn("local tool answered", self.xp)

    def test_xpchat_has_native_knowledge_context_injection(self):
        self.assertIn("augment_prompt_with_knowledge", self.xp)
        self.assertIn("Knowledge", self.xp)
        self.assertIn("Local knowledge snippets", self.xp)
        self.assertIn("*.txt", self.xp)
        self.assertIn("*.md", self.xp)
        self.assertIn("*.html", self.xp)
        self.assertIn("augment_prompt_with_knowledge(user_prompt)", self.xp)

    def test_backend_has_bounded_thread_summary_for_context_rollover(self):
        self.assertIn("thread_summary", self.nc)
        self.assertIn("Conversation so far:", self.nc)
        self.assertIn("bounded thread summary", self.nc)
        self.assertIn("summary_append_turn", self.nc)


if __name__ == "__main__":
    unittest.main()
