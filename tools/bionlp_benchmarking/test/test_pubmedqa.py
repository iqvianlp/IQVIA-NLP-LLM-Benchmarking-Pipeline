import unittest

from datasets import Dataset

from tools.bionlp_benchmarking.blurb.evaluators.pubmedqa import PubmedQAEvaluator


class PubmedQATestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.evaluator = PubmedQAEvaluator()

    def test_evaluate_all_no(self):
        predictions = {"dataset_name": "PubmedQA", "model_name": "TEST"}
        predictions["predictions"] = {k: "no" for k in self.evaluator._format_gold_standard(split="test").keys()}
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'PubmedQA', 'model_name': 'TEST', 'accuracy': 33.8}
        )

    def test_evaluate_all_yes(self):
        predictions = {"dataset_name": "PubmedQA", "model_name": "TEST"}
        predictions["predictions"] = {k: "yes" for k in self.evaluator._format_gold_standard(split="test").keys()}
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'PubmedQA', 'model_name': 'TEST', 'accuracy': 55.2}
        )

    def test_evaluate_all_correct(self):
        predictions = {"dataset_name": "PubmedQA", "model_name": "TEST"}
        predictions["predictions"] = {k: v[0] for k, v in self.evaluator._format_gold_standard(split="test").items()}
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'PubmedQA', 'model_name': 'TEST', 'accuracy': 100.0}
        )

    def test_evaluate_all_incorrect(self):
        predictions = {"dataset_name": "PubmedQA", "model_name": "TEST"}
        predictions["predictions"] = {
            k: "no" if v[0] == "yes" else "yes" for k, v in self.evaluator._format_gold_standard(split="test").items()
        }
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'PubmedQA', 'model_name': 'TEST', 'accuracy': 0.0}
        )

    def test_load_split(self):
        annotation_columns = ["answer"]
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
