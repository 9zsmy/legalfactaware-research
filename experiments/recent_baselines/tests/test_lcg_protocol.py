import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lcg_protocol import (
    acceptance_probability,
    exact_target_distance,
    extract_score_pair,
    objective_distance,
    objective_satisfied,
    select_final_chain,
    upper_budget_distance,
)


class LCGProtocolTests(unittest.TestCase):
    def test_upper_budget_distance(self):
        self.assertEqual(upper_budget_distance(450, 450), 0)
        self.assertEqual(upper_budget_distance(451, 450), 1)

    def test_exact_target_distance(self):
        self.assertEqual(exact_target_distance(430, 450), 20)
        self.assertEqual(exact_target_distance(450, 450), 0)
        self.assertEqual(exact_target_distance(470, 450), 20)

    def test_objective_helpers(self):
        self.assertTrue(objective_satisfied(449, 450, "upper_bound"))
        self.assertFalse(objective_satisfied(449, 450, "exact"))
        self.assertEqual(objective_distance(451, 450, "upper_bound"), 1)
        self.assertEqual(objective_distance(451, 450, "exact"), 1)

    def test_selects_first_successful_chain(self):
        self.assertEqual(select_final_chain([451, 450, 449], 450, "exact"), 1)
        self.assertEqual(select_final_chain([451, 430, 440], 450, "upper_bound"), 1)

    def test_selects_closest_chain_if_none_succeeds(self):
        self.assertEqual(select_final_chain([470, 455, 455], 450, "exact"), 1)
        self.assertEqual(select_final_chain([480, 451, 452], 450, "upper_bound"), 1)

    def test_rejects_empty_chain_list(self):
        with self.assertRaises(ValueError):
            select_final_chain([], 450, "exact")

    def test_satisfying_candidate_is_accepted(self):
        self.assertEqual(acceptance_probability(500, 450, 0.01, 450), 1.0)

    def test_length_improvement_changes_acceptance(self):
        self.assertEqual(acceptance_probability(500, 475, 1.0, 450), 1.0)
        self.assertAlmostEqual(acceptance_probability(475, 500, 1.0, 450), 0.5)

    def test_exact_acceptance_probability(self):
        self.assertEqual(
            acceptance_probability(200, 450, 0.1, 450, objective="exact"), 1.0
        )
        self.assertEqual(
            acceptance_probability(200, 400, 0.5, 450, objective="exact"), 1.0
        )
        self.assertAlmostEqual(
            acceptance_probability(400, 300, 1.0, 450, objective="exact"), 1 / 3
        )

    def test_json_score_extraction(self):
        self.assertEqual(
            extract_score_pair('{"summary_1_total": 42, "summary_2_total": 40}'),
            (42.0, 40.0),
        )


if __name__ == "__main__":
    unittest.main()

