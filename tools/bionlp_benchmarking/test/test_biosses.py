import unittest

from datasets import Dataset

from tools.bionlp_benchmarking.blurb.evaluators.biosses import BIOSSESEvaluator


class BIOSSESTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.evaluator = BIOSSESEvaluator()

    def test_evaluate_all_correct(self):
        predictions = {"dataset_name": "BIOSSES", "model_name": "TEST"}
        predictions["predictions"] = self.evaluator._format_gold_standard(split="test")
        self.assertEqual(
            {'dataset_name': 'BIOSSES', 'model_name': 'TEST', 'pearson': 1.0},
            self.evaluator.evaluate(predictions=predictions)
        )

    def test_evaluate_all_incorrect(self):
        predictions = {"dataset_name": "BIOSSES", "model_name": "TEST"}
        predictions["predictions"] = {
            k: v - 1.0 if v >= 2.0 else v + 1.0 for k, v in self.evaluator._format_gold_standard(split="test").items()
        }
        self.assertEqual(
            {'dataset_name': 'BIOSSES', 'model_name': 'TEST', 'pearson': 0.601},
            self.evaluator.evaluate(predictions=predictions)
        )

    def test_load_split(self):
        annotation_columns = ["annotator_a", "annotator_b", "annotator_c", "annotator_d", "annotator_e"]
        for split in self.evaluator.get_split_names():
            self.assertIsInstance(self.evaluator.load_split(split, as_dataset=False), dict)
            self.assertIsInstance(self.evaluator.load_split(split, as_dataset=True), Dataset)
            [
                [
                    self.assertNotIn(col, v) for col in annotation_columns
                ]
                for v in self.evaluator.load_split(split, as_dataset=False, unannotated=True).values()
            ]
            [
                [
                    self.assertIn(col, v) for col in annotation_columns
                ]
                for v in self.evaluator.load_split(split, as_dataset=False, unannotated=False).values()
            ]
