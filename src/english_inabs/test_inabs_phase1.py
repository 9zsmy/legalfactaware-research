import unittest

import numpy as np

import run_inabs_phase1 as phase1


class Phase1Tests(unittest.TestCase):
    def test_fixed_split_is_order_invariant(self):
        ids = [str(i) for i in range(20)]
        left = phase1.fixed_split(ids, seed=42, dev_cases=5)
        right = phase1.fixed_split(list(reversed(ids)), seed=42, dev_cases=5)
        self.assertEqual(left, right)
        self.assertEqual(len(left[0]), 15)
        self.assertEqual(len(left[1]), 5)

    def test_nearest_rank(self):
        values = list(range(1, 101))
        self.assertEqual(phase1.nearest_rank(values, 0.25), 25)
        self.assertEqual(phase1.nearest_rank(values, 0.50), 50)
        self.assertEqual(phase1.nearest_rank(values, 0.75), 75)

    def test_complete_sentence_budget_packing(self):
        sentences = ["one two three", "four five", "six seven eight nine"]
        self.assertEqual(phase1.pack_ranked(sentences, [2, 0, 1], 6), [1, 2])
        summary = phase1.summary(sentences, [1, 2])
        self.assertEqual(len(phase1.word_tokens(summary)), 6)

    def test_oracle_prefers_reference_sentence(self):
        sentences = ["irrelevant words here", "court allowed the appeal", "more unrelated text"]
        sequence, score, prefiltered = phase1.greedy_oracle_sequence(
            sentences, "the court allowed the appeal", budget=10
        )
        self.assertEqual(sequence[0], 1)
        self.assertGreater(score, 1.0)
        self.assertFalse(prefiltered)

    def test_lexrank_shape(self):
        scores = phase1.lexrank_scores(["alpha beta", "alpha beta gamma", "unrelated words"])
        self.assertEqual(scores.shape, (3,))
        self.assertTrue(np.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()
