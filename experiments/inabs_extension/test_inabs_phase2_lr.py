import unittest

import run_inabs_phase2_lr as phase2


class Phase2Tests(unittest.TestCase):
    def test_candidate_sampling(self):
        self.assertEqual(phase2.candidate_indices(3), [0, 1, 2])
        sampled = phase2.candidate_indices(1000)
        self.assertEqual(len(sampled), 400)
        self.assertEqual(sampled[0], 0)
        self.assertEqual(sampled[-1], 999)
        self.assertEqual(sampled, sorted(set(sampled)))

    def test_features(self):
        sentence = "On 12 January 2020, Rs. 500 was awarded under Section 14 of the Act."
        self.assertEqual(len(phase2.legal14(sentence, 2, 10)), 14)
        values = phase2.explicit20(sentence, 2, 10)
        self.assertEqual(len(values), 20)
        self.assertEqual(sum(values[14:]), 1.0)

    def test_decoder_respects_complete_sentence_budget(self):
        sentences = ["one two three", "four five", "six seven eight nine"]
        selected = phase2.decode(sentences, [0, 1, 2], phase2.np.array([0.9, 0.8, 0.7]), 6)
        self.assertLessEqual(sum(len(phase2.phase1.word_tokens(sentences[i])) for i in selected), 6)


if __name__ == "__main__":
    unittest.main()
