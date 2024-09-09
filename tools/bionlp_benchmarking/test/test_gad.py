import unittest

from datasets import Dataset

from tools.bionlp_benchmarking.blurb.evaluators.gad import GADEvaluator


class GADTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.evaluator = GADEvaluator()

    def test_evaluate_all_1(self):
        # all "1"
        predictions = {"dataset_name": "GAD", "model_name": "TEST"}
        predictions["predictions"] = {k: "1" for k in self.evaluator._format_gold_standard(split="test").keys()}
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'GAD', 'model_name': 'TEST', 'micro_f1': 52.62}
        )

    def test_evaluate_all_0(self):
        # all "0"
        predictions = {"dataset_name": "GAD", "model_name": "TEST"}
        predictions["predictions"] = {k: "0" for k in self.evaluator._format_gold_standard(split="test").keys()}
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'GAD', 'model_name': 'TEST', 'micro_f1': 47.38}
        )

    def test_evaluate_all_correct(self):
        # all correct
        predictions = {"dataset_name": "GAD", "model_name": "TEST"}
        predictions["predictions"] = self.evaluator._format_gold_standard(split="test")
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'GAD', 'model_name': 'TEST', 'micro_f1': 100.0}
        )

    def test_evaluate_all_incorrect(self):
        # all incorrect
        predictions = {"dataset_name": "GAD", "model_name": "TEST"}
        predictions["predictions"] = {
            k: "0" if v != "0" else "1" for k, v in
            self.evaluator._format_gold_standard(split="test").items()
        }
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'GAD', 'model_name': 'TEST', 'micro_f1': 0.0}
        )

    def test_load_split(self):
        annotation_columns = ["label"]
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
