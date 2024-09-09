import unittest

from datasets import Dataset

from tools.bionlp_benchmarking.blurb.evaluators.bioasq import BioASQEvaluator


class BioASQTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.evaluator = BioASQEvaluator()

    def test_evaluate_all_no(self):
        # all "no"
        predictions = {"dataset_name": "BioASQ", "model_name": "TEST"}
        predictions["predictions"] = {k: "no" for k in self.evaluator._format_gold_standard(split="test").keys()}
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'BioASQ', 'model_name': 'TEST', 'accuracy': 32.86}
        )

    def test_evaluate_all_yes(self):
        # all "yes"
        predictions = {"dataset_name": "BioASQ", "model_name": "TEST"}
        predictions["predictions"] = {k: "yes" for k in self.evaluator._format_gold_standard(split="test").keys()}
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'BioASQ', 'model_name': 'TEST', 'accuracy': 67.14}
        )

    def test_evaluate_all_correct(self):
        # all correct
        predictions = {"dataset_name": "BioASQ", "model_name": "TEST"}
        predictions["predictions"] = self.evaluator._format_gold_standard(split="test")
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'BioASQ', 'model_name': 'TEST', 'accuracy': 100.0}
        )

    def test_evaluate_all_incorrect(self):
        # all incorrect
        predictions = {"dataset_name": "BioASQ", "model_name": "TEST"}
        predictions["predictions"] = {
            k: "no" if v == "yes" else "yes" for k, v in self.evaluator._format_gold_standard(split="test").items()
        }
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'BioASQ', 'model_name': 'TEST', 'accuracy': 0.0}
        )

    def test_load_split(self):
        annotation_columns = ["concepts", "ideal_answer", "exact_answer", "triples"]
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
