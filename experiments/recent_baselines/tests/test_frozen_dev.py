#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_cail_baselines import evaluate_method
from protocol import read_jsonl


class FrozenDevTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_path = ROOT / "data" / "cail_dev_942.jsonl"
        cls.prediction_path = ROOT / "data" / "legalfactaware_predictions.jsonl"
        if not cls.dataset_path.exists() or not cls.prediction_path.exists():
            raise unittest.SkipTest("run prepare_cail_dev.py first")

    def test_frozen_counts_and_known_metrics(self):
        dataset = read_jsonl(self.dataset_path)
        predictions = read_jsonl(self.prediction_path)
        summary, _ = evaluate_method(dataset, predictions, "LegalFactAware-v2.5-B", 450)
        self.assertEqual(summary["cases"], 942)
        self.assertEqual(summary["budget_violation_count"], 0)
        self.assertAlmostEqual(summary["mean_characters"], 396.48513800424627, places=9)
        self.assertAlmostEqual(summary["expression_f1_document_macro"], 0.4785256022622569, places=9)


if __name__ == "__main__":
    unittest.main()

