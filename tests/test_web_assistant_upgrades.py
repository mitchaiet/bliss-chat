import tempfile
import unittest
from pathlib import Path

from server.bliss_xp_web_chat import answer_tool_prompt, augment_prompt_with_knowledge


class WebAssistantUpgradeTests(unittest.TestCase):
    def test_calculator_tool_answers_simple_arithmetic_without_model(self):
        result = answer_tool_prompt("calculate 12 * (3 + 4)")
        self.assertIsNotNone(result)
        self.assertEqual(result["answer"], "84")
        self.assertEqual(result["tool"], "calculator")

    def test_calculator_rejects_unsafe_expressions(self):
        self.assertIsNone(answer_tool_prompt("calculate __import__('os').system('dir')"))

    def test_knowledge_folder_adds_relevant_snippet_to_prompt(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "pc.txt").write_text("The Bliss target laptop is a Windows XP Home machine with Microsoft Sam.", encoding="utf-8")
            (root / "other.txt").write_text("Bananas are yellow fruit.", encoding="utf-8")
            shaped, hits = augment_prompt_with_knowledge("What voice does the XP laptop use?", root)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["file"], "pc.txt")
        self.assertIn("Microsoft Sam", shaped)
        # v1.3.0 contract: single-line "Context: <snippets> <question>" so the
        # one-line stdin protocol survives and the shape matches the v2
        # curated training format.
        self.assertTrue(shaped.startswith("Context: "))
        self.assertTrue(shaped.endswith("What voice does the XP laptop use?"))
        self.assertNotIn("\n", shaped)


if __name__ == "__main__":
    unittest.main()
