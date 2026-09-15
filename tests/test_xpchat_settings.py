"""Bounded compatibility checks using actual GUI helpers and a fake registry.

Run: python3 tests/test_xpchat_settings.py -v
No Windows registry, backend process, or model is accessed.
"""

from pathlib import Path
import unittest

import test_xpchat_transport as transport

ROOT = Path(__file__).resolve().parents[1]


class XpChatSettingsTests(unittest.TestCase):
    compile_and_run = transport.XpChatTransportTests.compile_and_run

    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "src" / "xpchat.c").read_text(encoding="utf-8")

    def test_saved_settings_and_model_banner_helpers(self):
        conversion = transport.excerpt(self.source, "static char *wide_to_utf8(",
                                       "static char *window_text_utf8(")
        self.assertEqual(conversion.count("wcslen(text)"), 1)
        self.compile_and_run("xpchat_settings_harness", {
            "/* GUI_DEFAULTS_SOURCE */": transport.excerpt(
                self.source, "#define DEFAULT_TEMP", "#define WM_APPEND_TEXT"),
            "/* GUI_WIDE_CONVERSION_SOURCE */": conversion.replace("wcslen(text)", "u16len(text)"),
            "/* GUI_SETTINGS_SOURCE */": transport.excerpt(
                self.source, "#define BLISS_REG_SETTINGS", "static void settings_save("),
            "/* GUI_MODEL_NAME_SOURCE */": transport.excerpt(
                self.source, "static const char *backend_model_name(", "static LRESULT CALLBACK wnd_proc("),
        }, "GUI_SETTINGS_COMPATIBILITY passed")

    def test_context_is_fixed_before_process_creation(self):
        start = self.source.index('"\\\"%s\\\\%s\\\" \\\"%s\\\\%s\\\" \\\"%s\\\\%s\\\" -c 512')
        end = self.source.index("if (!CreateProcessA(", start)
        command = self.source[start:end]
        self.assertIn("-c 512 -t 0.0 -p 0.95", command)
        self.assertNotIn("RegQuery", command)
        self.assertNotIn("gMaxTok", command)
        settings = transport.excerpt(self.source, "static void settings_load(", "static void settings_save(")
        self.assertNotIn('"Context"', settings)
        self.assertNotIn('"Ctx"', settings)

    def test_settings_are_normalized_before_save_and_send(self):
        save = transport.excerpt(self.source, "static void settings_save(", "static void settings_reset_all(")
        apply = transport.excerpt(self.source, "static void settings_apply_to_backend(", "static void apply_settings_preset(")
        self.assertLess(save.index("settings_normalize();"), save.index("RegSetValueExA("))
        self.assertLess(apply.index("settings_normalize();"), apply.index("backend_send_line("))

    def test_about_and_model_info_use_reported_identity(self):
        about = self.source[self.source.index("case IDM_ABOUT: {"):self.source.index("case IDM_EXIT:")]
        info = self.source[self.source.index("case IDM_MODELINFO: {"):self.source.index("case IDM_PERF:")]
        for dialog in (about, info):
            self.assertIn("gModelName, gModelSource", dialog)
            self.assertIn("message_box_utf8(", dialog)
            self.assertNotIn("SmolLM2-360M", dialog)
        events = self.source[self.source.index("case WM_BACKEND_INFO: {"):self.source.index("case WM_BACKEND_DEAD:")]
        self.assertIn('"SOURCE "', events)
        self.assertIn("copy_utf8_prefix(gModelSource", events)


if __name__ == "__main__":
    unittest.main()
