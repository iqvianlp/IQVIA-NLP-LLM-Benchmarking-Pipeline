import unittest

from datasets import Dataset

from tools.bionlp_benchmarking.blurb.evaluators.ner import NEREvaluator


class NERTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.evaluators = {
            "JNLPBA": NEREvaluator("JNLPBA"),
            "BC5-chem": NEREvaluator("BC5-chem"),
            "BC5-disease": NEREvaluator("BC5-disease"),
            "NCBI-disease": NEREvaluator("NCBI-disease"),
            "BC2GM": NEREvaluator("BC2GM")
        }

    def test_evaluate_no_entities(self):
        expected = {
            "JNLPBA": {
                'dataset_name': 'JNLPBA',
                'model_name': 'TEST',
                'exact': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'partial': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'strict': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'type': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
            },
            "BC5-chem": {
                'dataset_name': 'BC5-chem',
                'model_name': 'TEST',
                'exact': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'partial': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'strict': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'type': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
            },
            "BC5-disease": {
                'dataset_name': 'BC5-disease',
                'model_name': 'TEST',
                'exact': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'partial': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'strict': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'type': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
            },
            "NCBI-disease": {
                'dataset_name': 'NCBI-disease',
                'model_name': 'TEST',
                'exact': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'partial': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'strict': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'type': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
            },
            "BC2GM": {
                'dataset_name': 'BC2GM',
                'model_name': 'TEST',
                'exact': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'partial': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'strict': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'type': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
            }
        }
        for name in self.evaluators:
            predictions = {"dataset_name": name, "model_name": "TEST"}
            predictions["predictions"] = {}
            for item in self.evaluators[name]._dataset["test"]:
                predictions["predictions"][item["id"]] = ["O"] * len(item["ner_tags"])
            self.assertEqual(self.evaluators[name].evaluate(predictions=predictions), expected[name])

    def test_evaluate_all_false_positives(self):
        expected = {
            "JNLPBA": {
                'dataset_name': 'JNLPBA',
                'model_name': 'TEST',
                'exact': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'partial': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'strict': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'type': {'f1': 19.1, 'precision': 10.56, 'recall': 100.0}
            },
            "BC5-chem": {
                'dataset_name': 'BC5-chem',
                'model_name': 'TEST',
                'exact': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'partial': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'strict': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'type': {'f1': 8.75, 'precision': 4.57, 'recall': 100.0}
            },
            "BC5-disease": {
                'dataset_name': 'BC5-disease',
                'model_name': 'TEST',
                'exact': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'partial': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'strict': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'type': {'f1': 7.25, 'precision': 3.76, 'recall': 100.0}
            },
            "NCBI-disease": {
                'dataset_name': 'NCBI-disease',
                'model_name': 'TEST',
                'exact': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'partial': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'strict': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'type': {'f1': 8.2, 'precision': 4.28, 'recall': 100.0}
            },
            "BC2GM": {
                'dataset_name': 'BC2GM',
                'model_name': 'TEST',
                'exact': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'partial': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'strict': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'type': {'f1': 9.39, 'precision': 4.93, 'recall': 100.0}
            }
        }
        m = {0: "O", 1: "B", 2: "I"}
        for name in self.evaluators:
            predictions = {"dataset_name": name, "model_name": "TEST"}
            predictions["predictions"] = {}
            for item in self.evaluators[name]._dataset["test"]:
                predictions["predictions"][item["id"]] = \
                    ["O" if other_i in ["B", "I"] else "B" for other_i in [m[i] for i in item["ner_tags"]]]
            self.assertEqual(self.evaluators[name].evaluate(predictions=predictions), expected[name])

    def test_evaluate_all_partial_matches(self):
        expected = {
            "JNLPBA": {
                'dataset_name': 'JNLPBA',
                'model_name': 'TEST',
                'exact': {'f1': 40.01, 'precision': 40.01, 'recall': 40.01},
                'partial': {'f1': 100.0, 'precision': 100.0, 'recall': 100.0},
                'strict': {'f1': 40.01, 'precision': 40.01, 'recall': 40.01},
                'type': {'f1': 100.0, 'precision': 100.0, 'recall': 100.0}
            },
            "BC5-chem": {
                'dataset_name': 'BC5-chem',
                'model_name': 'TEST',
                'exact': {'f1': 88.04, 'precision': 88.04, 'recall': 88.04},
                'partial': {'f1': 100.0, 'precision': 100.0, 'recall': 100.0},
                'strict': {'f1': 88.04, 'precision': 88.04, 'recall': 88.04},
                'type': {'f1': 100.0, 'precision': 100.0, 'recall': 100.0}
            },
            "BC5-disease": {
                'dataset_name': 'BC5-disease',
                'model_name': 'TEST',
                'exact': {'f1': 60.26, 'precision': 60.26, 'recall': 60.26},
                'partial': {'f1': 100.0, 'precision': 100.0, 'recall': 100.0},
                'strict': {'f1': 60.26, 'precision': 60.26, 'recall': 60.26},
                'type': {'f1': 100.0, 'precision': 100.0, 'recall': 100.0}
            },
            "NCBI-disease": {
                'dataset_name': 'NCBI-disease',
                'model_name': 'TEST',
                'exact': {'f1': 44.06, 'precision': 44.06, 'recall': 44.06},
                'partial': {'f1': 100.0, 'precision': 100.0, 'recall': 100.0},
                'strict': {'f1': 44.06, 'precision': 44.06, 'recall': 44.06},
                'type': {'f1': 100.0, 'precision': 100.0, 'recall': 100.0}
            },
            "BC2GM": {
                'dataset_name': 'BC2GM',
                'model_name': 'TEST',
                'exact': {'f1': 44.62, 'precision': 44.62, 'recall': 44.62},
                'partial': {'f1': 100.0, 'precision': 100.0, 'recall': 100.0},
                'strict': {'f1': 44.62, 'precision': 44.62, 'recall': 44.62},
                'type': {'f1': 100.0, 'precision': 100.0, 'recall': 100.0}
            }
        }
        m = {0: "O", 1: "B", 2: "I"}
        for name in self.evaluators:
            predictions = {"dataset_name": name, "model_name": "TEST"}
            predictions["predictions"] = {}
            for item in self.evaluators[name]._dataset["test"]:
                predictions["predictions"][item["id"]] = \
                    ["O" if other_i == "I" else other_i for other_i in [m[i] for i in item["ner_tags"]]]
            self.assertEqual(self.evaluators[name].evaluate(predictions=predictions), expected[name])

    def test_evaluate_all_exact_matches(self):
        expected = {
            "JNLPBA": {
                'dataset_name': 'JNLPBA',
                'model_name': 'TEST',
                'exact': {'f1': 100.0, 'precision': 100.0, 'recall': 100.0},
                'partial': {'f1': 100.0, 'precision': 100.0, 'recall': 100.0},
                'strict': {'f1': 100.0, 'precision': 100.0, 'recall': 100.0},
                'type': {'f1': 100.0, 'precision': 100.0, 'recall': 100.0}
            },
            "BC5-chem": {
                'dataset_name': 'BC5-chem',
                'model_name': 'TEST',
                'exact': {'f1': 100.0, 'precision': 100.0, 'recall': 100.0},
                'partial': {'f1': 100.0, 'precision': 100.0, 'recall': 100.0},
                'strict': {'f1': 100.0, 'precision': 100.0, 'recall': 100.0},
                'type': {'f1': 100.0, 'precision': 100.0, 'recall': 100.0}
            },
            "BC5-disease": {
                'dataset_name': 'BC5-disease',
                'model_name': 'TEST',
                'exact': {'f1': 100.0, 'precision': 100.0, 'recall': 100.0},
                'partial': {'f1': 100.0, 'precision': 100.0, 'recall': 100.0},
                'strict': {'f1': 100.0, 'precision': 100.0, 'recall': 100.0},
                'type': {'f1': 100.0, 'precision': 100.0, 'recall': 100.0}
            },
            "NCBI-disease": {
                'dataset_name': 'NCBI-disease',
                'model_name': 'TEST',
                'exact': {'f1': 100.0, 'precision': 100.0, 'recall': 100.0},
                'partial': {'f1': 100.0, 'precision': 100.0, 'recall': 100.0},
                'strict': {'f1': 100.0, 'precision': 100.0, 'recall': 100.0},
                'type': {'f1': 100.0, 'precision': 100.0, 'recall': 100.0}
            },
            "BC2GM": {
                'dataset_name': 'BC2GM',
                'model_name': 'TEST',
                'exact': {'f1': 100.0, 'precision': 100.0, 'recall': 100.0},
                'partial': {'f1': 100.0, 'precision': 100.0, 'recall': 100.0},
                'strict': {'f1': 100.0, 'precision': 100.0, 'recall': 100.0},
                'type': {'f1': 100.0, 'precision': 100.0, 'recall': 100.0}
            }
        }
        m = {0: "O", 1: "B", 2: "I"}
        for name in self.evaluators:
            predictions = {"dataset_name": name, "model_name": "TEST"}
            predictions["predictions"] = {}
            for item in self.evaluators[name]._dataset["test"]:
                predictions["predictions"][item["id"]] = [m[i] for i in item["ner_tags"]]
            self.assertEqual(self.evaluators[name].evaluate(predictions=predictions), expected[name])

    def test_load_split(self):
        annotation_columns = ["type", "ner_tags"]
        for name in self.evaluators:
            for split in self.evaluators[name].get_split_names():
                self.assertIsInstance(self.evaluators[name].load_split(split, as_dataset=False), dict)
                self.assertIsInstance(self.evaluators[name].load_split(split, as_dataset=True), Dataset)
                [
                    [
                        self.assertNotIn(col, v) for col in annotation_columns
                    ]
                    for v in self.evaluators[name].load_split(split, as_dataset=False, unannotated=True).values()
                ]
                [
                    [
                        self.assertIn(col, v) for col in annotation_columns
                    ]
                    for v in self.evaluators[name].load_split(split, as_dataset=False, unannotated=False).values()
                ]
