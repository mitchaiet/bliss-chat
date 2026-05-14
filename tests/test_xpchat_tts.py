import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
XPCHAT = ROOT / "src" / "xpchat.c"
BUILD = ROOT / "scripts" / "build-xp.sh"


def read_xpchat() -> str:
    return XPCHAT.read_text(encoding="utf-8")


class XpChatTtsTests(unittest.TestCase):
    def test_microsoft_sam_speak_button_is_wired_to_last_reply(self):
        src = read_xpchat()
        self.assertIn("IDC_SPEAK_LAST", src)
        self.assertIn("Speak last reply", src)
        self.assertIn("speak_last_reply", src)
        self.assertIn("rich_get_range(gLastAsstStart, gLastAsstEnd)", src)

    def test_tts_uses_xp_sapi_spvoice_without_modern_dependencies(self):
        src = read_xpchat()
        self.assertIn('CLSIDFromProgID(L"SAPI.SpVoice"', src)
        self.assertIn('GetIDsOfNames', src)
        self.assertIn('Invoke', src)
        self.assertIn('L"Speak"', src)
        self.assertIn("COINIT_APARTMENTTHREADED", src)

    def test_build_links_ole_libraries_for_sapi_automation(self):
        build = BUILD.read_text(encoding="utf-8")
        self.assertIn("-lole32", build)
        self.assertIn("-loleaut32", build)
        self.assertIn("-luuid", build)

    def test_send_button_always_uses_owner_drawn_green_arrow(self):
        src = read_xpchat()
        self.assertIn("BS_DEFPUSHBUTTON | BS_OWNERDRAW", src)
        self.assertIn("draw_command_button", src)
        self.assertIn("dis->CtlID == IDC_SEND", src)
        self.assertIn("RGB(31, 168, 38)", src)
        self.assertNotIn("send_icons[]", src)


if __name__ == "__main__":
    unittest.main()
