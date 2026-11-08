import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zero_shot_lc_protocol import (
    count_characters,
    is_compliant,
    revision_feedback,
    select_candidate,
    selection_key,
)


class ZeroShotLCProtocolTests(unittest.TestCase):
    def test_character_count_matches_python_len(self):
        self.assertEqual(count_characters("甲 乙。"), 4)

    def test_exact_sample_filtering(self):
        self.assertEqual(select_candidate([420, 452, 470], 450, "exact"), 1)

    def test_exact_tie_uses_generation_order(self):
        self.assertEqual(select_candidate([440, 460], 450, "exact"), 0)

    def test_upper_bound_prefers_compliant_candidate(self):
        self.assertEqual(select_candidate([451, 430, 440], 450, "upper_bound"), 2)

    def test_upper_bound_prefers_smallest_overflow_if_needed(self):
        self.assertEqual(select_candidate([470, 451, 460], 450, "upper_bound"), 1)

    def test_paper_tolerance_is_inclusive(self):
        self.assertTrue(is_compliant(405, 450, "exact", 0.10))
        self.assertTrue(is_compliant(495, 450, "exact", 0.10))
        self.assertFalse(is_compliant(404, 450, "exact", 0.10))

    def test_hard_upper_bound(self):
        self.assertTrue(is_compliant(1, 450, "upper_bound"))
        self.assertTrue(is_compliant(450, 450, "upper_bound"))
        self.assertFalse(is_compliant(451, 450, "upper_bound"))

    def test_invalid_objective_fails(self):
        with self.assertRaises(ValueError):
            selection_key(450, 450, "invalid")

    def test_feedback_contains_measured_difference(self):
        feedback = revision_feedback(480, 450, "exact")
        self.assertIn("480", feedback)
        self.assertIn("多30", feedback)
        self.assertIn("450", feedback)


if __name__ == "__main__":
    unittest.main()

