import unittest

from datasets import Dataset

from tools.bionlp_benchmarking.blurb.evaluators.ddi import DDIEvaluator


class DDITestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.evaluator = DDIEvaluator()

    def test_evaluate_all_ddi_effect(self):
        predictions = {"dataset_name": "DDI", "model_name": "TEST"}
        predictions["predictions"] = {
            k: "DDI-effect" for k in self.evaluator._format_gold_standard(split="test").keys()
        }
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'DDI', 'model_name': 'TEST', 'micro_f1': 10.75}
        )

    def test_evaluate_all_ddi_int(self):
        predictions = {"dataset_name": "DDI", "model_name": "TEST"}
        predictions["predictions"] = {
            k: "DDI-int" for k in self.evaluator._format_gold_standard(split="test").keys()
        }
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'DDI', 'model_name': 'TEST', 'micro_f1': 2.87}
        )

    def test_evaluate_all_ddi_mechanism(self):
        predictions = {"dataset_name": "DDI", "model_name": "TEST"}
        predictions["predictions"] = {
            k: "DDI-mechanism" for k in self.evaluator._format_gold_standard(split="test").keys()
        }
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'DDI', 'model_name': 'TEST', 'micro_f1': 9.02}
        )

    def test_evaluate_all_ddi_advise(self):
        predictions = {"dataset_name": "DDI", "model_name": "TEST"}
        predictions["predictions"] = {
            k: "DDI-advise" for k in self.evaluator._format_gold_standard(split="test").keys()
        }
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'DDI', 'model_name': 'TEST', 'micro_f1': 6.6}
        )

    def test_evaluate_all_correct(self):
        predictions = {"dataset_name": "DDI", "model_name": "TEST"}
        predictions["predictions"] = self.evaluator._format_gold_standard(split="test")
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'DDI', 'model_name': 'TEST', 'micro_f1': 100.0}
        )

    def test_evaluate_all_incorrect(self):
        predictions = {"dataset_name": "DDI", "model_name": "TEST"}
        predictions["predictions"] = {
            k: "DDI-effect" if v != "DDI-effect" else "DDI-advise" for k, v in
            self.evaluator._format_gold_standard(split="test").items()
        }
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'DDI', 'model_name': 'TEST', 'micro_f1': 0.0}
        )

    def test_load_split(self):
        annotation_columns = ["relations"]
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

