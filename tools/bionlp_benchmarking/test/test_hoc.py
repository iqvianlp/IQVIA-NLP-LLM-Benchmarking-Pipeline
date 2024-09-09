import unittest

from datasets import Dataset

from tools.bionlp_benchmarking.blurb.evaluators.hoc import HoCEvaluator


class HoCTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.evaluator = HoCEvaluator()

    def test_evaluate_gs(self):
        predictions = {"dataset_name": "HoC", "model_name": "TEST"}
        predictions["predictions"] = {
            k: ["GS"] * len(v) for k, v in self.evaluator._format_gold_standard(split="test").items()
        }
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'HoC', 'model_name': 'TEST', 'average_micro_f1': 2.86}
        )

    def test_evaluate_tpi(self):
        predictions = {"dataset_name": "HoC", "model_name": "TEST"}
        predictions["predictions"] = {
            k: ["TPI"] * len(v) for k, v in self.evaluator._format_gold_standard(split="test").items()
        }
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'HoC', 'model_name': 'TEST', 'average_micro_f1': 2.23}
        )

    def test_evaluate_ri(self):
        predictions = {"dataset_name": "HoC", "model_name": "TEST"}
        predictions["predictions"] = {
            k: ["RI"] * len(v) for k, v in self.evaluator._format_gold_standard(split="test").items()
        }
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'HoC', 'model_name': 'TEST', 'average_micro_f1': 1.4}
        )

    def test_evaluate_ce(self):
        predictions = {"dataset_name": "HoC", "model_name": "TEST"}
        predictions["predictions"] = {
            k: ["CE"] * len(v) for k, v in self.evaluator._format_gold_standard(split="test").items()
        }
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'HoC', 'model_name': 'TEST', 'average_micro_f1': 1.13}
        )

    def test_evaluate_cd(self):
        predictions = {"dataset_name": "HoC", "model_name": "TEST"}
        predictions["predictions"] = {
            k: ["CD"] * len(v) for k, v in self.evaluator._format_gold_standard(split="test").items()
        }
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'HoC', 'model_name': 'TEST', 'average_micro_f1': 3.98}
        )

    def test_evaluate_im(self):
        predictions = {"dataset_name": "HoC", "model_name": "TEST"}
        predictions["predictions"] = {
            k: ["IM"] * len(v) for k, v in self.evaluator._format_gold_standard(split="test").items()
        }
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'HoC', 'model_name': 'TEST', 'average_micro_f1': 3.26}
        )

    def test_evaluate_gi(self):
        predictions = {"dataset_name": "HoC", "model_name": "TEST"}
        predictions["predictions"] = {
            k: ["GI"] * len(v) for k, v in self.evaluator._format_gold_standard(split="test").items()
        }
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'HoC', 'model_name': 'TEST', 'average_micro_f1': 3.39}
        )

    def test_evaluate_a(self):
        predictions = {"dataset_name": "HoC", "model_name": "TEST"}
        predictions["predictions"] = {
            k: ["A"] if v else [] for k, v in self.evaluator._format_gold_standard(split="test").items()
        }
        print(predictions["predictions"])
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'HoC', 'model_name': 'TEST', 'average_micro_f1': 1.35}
        )

    def test_evaluate_ps(self):
        predictions = {"dataset_name": "HoC", "model_name": "TEST"}
        predictions["predictions"] = {
            k: ["PS"] * len(v) for k, v in self.evaluator._format_gold_standard(split="test").items()
        }
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'HoC', 'model_name': 'TEST', 'average_micro_f1': 4.96}
        )

    def test_evaluate_id(self):
        predictions = {"dataset_name": "HoC", "model_name": "TEST"}
        predictions["predictions"] = {
            k: ["ID"] * len(v) for k, v in self.evaluator._format_gold_standard(split="test").items()
        }
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'HoC', 'model_name': 'TEST', 'average_micro_f1': 0.84}
        )

    def test_evaluate_all_correct(self):
        predictions = {"dataset_name": "HoC", "model_name": "TEST"}
        predictions["predictions"] = self.evaluator._format_gold_standard(split="test")
        self.assertEqual(
            self.evaluator.evaluate(predictions=predictions),
            {'dataset_name': 'HoC', 'model_name': 'TEST', 'average_micro_f1': 100.0}
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
