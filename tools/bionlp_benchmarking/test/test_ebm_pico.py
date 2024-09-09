import unittest

from datasets import Dataset

from tools.bionlp_benchmarking.blurb.evaluators.ebm_pico import EBMPICOEvaluator


class GADTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.evaluator = EBMPICOEvaluator()

    def test_evaluate_i_int(self):
        # all "I-INT"
        predictions = {"dataset_name": "EBM PICO", "model_name": "TEST"}
        predictions["predictions"] = {
            k: ["I-INT"] * len(v) for k, v in self.evaluator._format_gold_standard(split="test").items()
        }
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'EBM PICO', 'model_name': 'TEST', 'macro_f1_word_level': 2.21}
        )

    def test_evaluate_i_out(self):
        # all "I-OUT"
        predictions = {"dataset_name": "EBM PICO", "model_name": "TEST"}
        predictions["predictions"] = {
            k: ["I-OUT"] * len(v) for k, v in self.evaluator._format_gold_standard(split="test").items()
        }
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'EBM PICO', 'model_name': 'TEST', 'macro_f1_word_level': 4.92}
        )

    def test_evaluate_i_par(self):
        # all "I-PAR"
        predictions = {"dataset_name": "EBM PICO", "model_name": "TEST"}
        predictions["predictions"] = {
            k: ["I-PAR"] * len(v) for k, v in self.evaluator._format_gold_standard(split="test").items()
        }
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'EBM PICO', 'model_name': 'TEST', 'macro_f1_word_level': 3.95}
        )

    def test_evaluate_all_correct(self):
        # all correct
        predictions = {"dataset_name": "EBM PICO", "model_name": "TEST"}
        predictions["predictions"] = self.evaluator._format_gold_standard(split="test")
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'EBM PICO', 'model_name': 'TEST', 'macro_f1_word_level': 100.0}
        )

    def test_evaluate_all_incorrect(self):
        # all incorrect
        predictions = {"dataset_name": "EBM PICO", "model_name": "TEST"}
        predictions["predictions"] = {
            k: ["I-INT" if i != "I-INT" else "O" for i in v] for k, v in
            self.evaluator._format_gold_standard(split="test").items()
        }
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'EBM PICO', 'model_name': 'TEST', 'macro_f1_word_level': 0.0}
        )

    def test_load_split(self):
        annotation_columns = ["entities"]
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
