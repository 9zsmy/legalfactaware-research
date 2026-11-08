#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from protocol import character_count, expression_sets, rouge_l, rouge_n, truncate_source


class ProtocolTests(unittest.TestCase):
    def test_exact_match_rouge(self):
        text = "原告请求被告支付100元。"
        self.assertEqual(rouge_n(text, text, 1), 1.0)
        self.assertEqual(rouge_n(text, text, 2), 1.0)
        self.assertEqual(rouge_l(text, text), 1.0)

    def test_empty_prediction(self):
        self.assertEqual(rouge_n("有效摘要", "", 1), 0.0)
        self.assertEqual(rouge_l("有效摘要", ""), 0.0)

    def test_head_tail_preserves_budget_and_ends(self):
        text = "0123456789"
        result = truncate_source(text, 6, "head_tail")
        self.assertEqual(result, "012789")
        self.assertEqual(len(result), 6)

    def test_character_count_matches_locked_python_len(self):
        self.assertEqual(character_count("甲 乙"), 3)

    def test_expression_types(self):
        values = expression_sets("本院于2020年1月2日判决被告支付人民币10万元，依照《合同法》第一条。")
        self.assertTrue(values["amount"])
        self.assertTrue(values["date"])
        self.assertTrue(values["law"])
        self.assertIn("支付", values["decision"])


if __name__ == "__main__":
    unittest.main()

