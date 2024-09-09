import unittest

from datasets import Dataset

from tools.bionlp_benchmarking.blurb.evaluators.chemprot import ChemProtEvaluator


class ChemProtTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.evaluator = ChemProtEvaluator()

    def test_evaluate_all_cpr3(self):
        predictions = {"dataset_name": "ChemProt", "model_name": "TEST"}
        predictions["predictions"] = {k: "CPR:3" for k in self.evaluator._format_gold_standard(split="test").keys()}
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'ChemProt', 'model_name': 'TEST', 'micro_f1': 6.92}
        )

    def test_evaluate_all_cpr4(self):
        predictions = {"dataset_name": "ChemProt", "model_name": "TEST"}
        predictions["predictions"] = {k: "CPR:4" for k in self.evaluator._format_gold_standard(split="test").keys()}
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'ChemProt', 'model_name': 'TEST', 'micro_f1': 17.26}
        )

    def test_evaluate_all_cpr5(self):
        predictions = {"dataset_name": "ChemProt", "model_name": "TEST"}
        predictions["predictions"] = {k: "CPR:5" for k in self.evaluator._format_gold_standard(split="test").keys()}
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'ChemProt', 'model_name': 'TEST', 'micro_f1': 1.86}
        )

    def test_evaluate_all_cpr6(self):
        predictions = {"dataset_name": "ChemProt", "model_name": "TEST"}
        predictions["predictions"] = {k: "CPR:6" for k in self.evaluator._format_gold_standard(split="test").keys()}
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'ChemProt', 'model_name': 'TEST', 'micro_f1': 3.05}
        )

    def test_evaluate_all_cpr9(self):
        predictions = {"dataset_name": "ChemProt", "model_name": "TEST"}
        predictions["predictions"] = {k: "CPR:9" for k in self.evaluator._format_gold_standard(split="test").keys()}
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'ChemProt', 'model_name': 'TEST', 'micro_f1': 6.7}
        )

    def test_evaluate_all_correct(self):
        predictions = {"dataset_name": "ChemProt", "model_name": "TEST"}
        predictions["predictions"] = self.evaluator._format_gold_standard(split="test")
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'ChemProt', 'model_name': 'TEST', 'micro_f1': 100.0}
        )

    def test_evaluate_all_incorrect(self):
        predictions = {"dataset_name": "ChemProt", "model_name": "TEST"}
        predictions["predictions"] = {
            k: "CPR:3" if v != "CPR:3" else "CPR:4" for k, v in
            self.evaluator._format_gold_standard(split="test").items()
        }
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'ChemProt', 'model_name': 'TEST', 'micro_f1': 0.0}
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


