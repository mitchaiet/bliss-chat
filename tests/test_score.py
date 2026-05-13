import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

from score import repetition_features, score_row


class ScoreDiagnosticsTests(unittest.TestCase):
    def test_repetition_features_detect_consecutive_token_loop(self):
        features = repetition_features("the the the the answer")

        self.assertEqual(features["max_token_run"], 4)
        self.assertEqual(features["ramble_reason"], "token_run")

    def test_repetition_features_detect_phrase_loop(self):
        features = repetition_features("Paris is France. Paris is France. Paris is France.")

        self.assertGreaterEqual(features["max_ngram_repeats"], 3)
        self.assertEqual(features["ramble_reason"], "ngram_repeat")

    def test_score_row_outputs_diagnostic_columns_for_rambles(self):
        row = {
            "id": "x",
            "category": "basic",
            "question": "Capital of France?",
            "expected_keywords": "paris",
            "response": "Paris Paris Paris Paris Paris",
        }

        scored = score_row(row)

        self.assertEqual(scored["coherent"], 0)
        self.assertEqual(scored["ramble"], 1)
        self.assertEqual(scored["ramble_reason"], "token_run")
        self.assertNotEqual(scored["unique_token_ratio"], "")


if __name__ == "__main__":
    unittest.main()
