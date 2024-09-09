from abc import abstractmethod
import os
from typing import List

from datasets import load_dataset
from jsonschema import validate
from sklearn.metrics import f1_score, precision_score, recall_score

from utils.io_utils import load_json


PROCESSED_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "resources", "processed"
)

SCHEMAS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "resources", "schemas"
)


class BLURBEvaluator(object):
    """
    Implementation of base object for the schema-specific BLURB evaluators

    Args:
        dataset_name: name of the dataset
        schema_path: path to the schema JSON file
        hf_dataset_path: path to a dataset on Hugging Face Hub
        hf_dataset_name: name of a dataset subset on Hugging Face hub
        data_dir: directory containing data used for Hugging Face datasets that require local files
        json_dataset_path: path to a local JSON-formatted dataset. If specified, the arguments `hf_dataset_path`,
            `hf_dataset_name`, and `data_dir` will be ignored.
    """
    def __init__(
        self,
        dataset_name: str,
        schema_path: str = None,
        hf_dataset_path: str = None,
        hf_dataset_name: str = None,
        data_dir: str = None,
        json_dataset_path: str = None
    ):
        self._dataset_name = dataset_name

        # default URLs and file paths
        if schema_path is None:
            schema_path = os.path.join(SCHEMAS_DIR, f"{self._dataset_name}.json")

        # store dataset splits as values in a dictionary; split names (e.g. train, test) are keys
        if json_dataset_path is not None:
            self._dataset = self._load_json_dataset(json_dataset_path)
        else:
            self._dataset = self._load_dataset(
                hf_dataset_path=hf_dataset_path,
                hf_dataset_name=hf_dataset_name,
                data_dir=data_dir
            )

        # store the prediction schema as dictionary
        self._schema = self._load_schema(
            schema_path=schema_path
        )

    def validate_predictions(self, predictions: dict) -> bool:
        """
        Validates prediction dictionary against the schema

        Parameters:
            predictions: dictionary containing the predictions to validate against the schema

        Returns:
            True if inputted dictionary of predictions satisfies the schema

        Raises:
            ValidationError
        """
        validate(instance=predictions, schema=self._schema)
        return True

    def get_description_dict(self):
        """
        Return a dictionary describing some general features of the distinct splits of the dataset
        :return:
        """

        def describe_data(input_el):
            if isinstance(input_el, dict):
                return {k: describe_data(v) for k, v in input_el.items()}
            elif isinstance(input_el, list):
                return {"type": "list", "element": describe_data(input_el[0]) if len(input_el) > 0 else "DATA"}
            elif isinstance(input_el, tuple):
                return {"type": "tuple", "element": describe_data(input_el[0]) if len(input_el) > 0 else "DATA"}
            elif isinstance(input_el, set):
                return {"type": "set", "element": describe_data(list(input_el)[0]) if len(input_el) > 0 else "DATA"}
            elif isinstance(input_el, str):
                return {"type": "str"}
            elif isinstance(input_el, int):
                return {"type": "int"}
            elif isinstance(input_el, float):
                return {"type": "float"}
            elif isinstance(input_el, bool):
                return {"type": "bool"}
            else:
                return "DATA"

        dataset_info_dict = dict()
        dataset_info_dict["split_list"] = self.get_split_names()
        dataset_info_dict["split_info"] = dict()
        for split in self.get_split_names():
            info_dict = dict()
            info_dict["length"] = len(self._dataset[split])
            info_dict["item_description"] = describe_data(self._dataset[split][0])

            dataset_info_dict["split_info"][split] = info_dict

        return dataset_info_dict

    def get_split_names(self):
        """
        Get list of available split names
        :return: list of available split names
        """
        return [split_name for split_name in self._dataset]

    @abstractmethod
    def load_split(self, split: str, unannotated: bool = False, as_dataset: bool = False):
        pass

    @staticmethod
    def _load_dataset(
            hf_dataset_path: str,
            hf_dataset_name: str,
            data_dir: str = None
    ) -> dict:
        """
        Loads dataset from Hugging Face Hub

        Args:
            hf_dataset_path: path to a dataset on Hugging Face Hub
            hf_dataset_name: name of a dataset subset on Hugging Face hub
            data_dir: directory containing data used for Hugging Face datasets that require local files

        Returns:
            Dataset object
        """
        dataset = load_dataset(hf_dataset_path, hf_dataset_name, data_dir=data_dir, trust_remote_code=True)
        output = {split: dataset[split] for split in dataset}
        return output

    @staticmethod
    def _load_json_dataset(json_dataset_path: str) -> dict:
        """
        Loads dataset from local JSON file.

        Args:
            json_dataset_path: Path to the JSON-formatted dataset.

        Returns:
            Dataset object
        """
        dataset = load_dataset(json_dataset_path)
        output = {split: dataset[split] for split in dataset}
        for split in dataset:
            if split == 'validation':
                output['dev'] = dataset[split]
            else:
                output[split] = dataset[split]
        return output

    @staticmethod
    def _load_schema(schema_path: str) -> dict:
        """
        Load JSON schema from local source for the first time; then read from cache

        Args:
            schema_path: path to store the JSON schema locally

        Returns:
            dictionary containing the schema
        """
        return load_json(fp=schema_path)


class NERMetricComputation(object):
    """
    Implementation of precision, recall, and F1 computation for named entity recognition tasks

    Args:
        gold_standard: list of lists; each inner list corresponds to a document; for each annotated document, the
            corresponding list contains dictionaries that hold gold standard annotation information in the format of:
            {"label": the type of entity, "start": span start, "end": span end}

        predictions: list of lists formatted like 'gold_standards'; inner lists that correspond to the same document
            (across 'gold_standard' and 'predictions') must be at the same index of the outer list
    """
    def __init__(self, gold_standard: list, predictions: list):
        self._gold_standard = gold_standard
        self._predictions = predictions

    def compute(self, strictness: str = "strict") -> dict:
        """
        computes precision, recall and F1

        Args:
            strictness: one of 'strict', 'exact', 'partial', or 'type'
        """
        def compare_doc_entities(partial: bool = False):
            """
            compare the list of gold standard entities and the list of predicted entities for a document
            """
            # infrastructure for tracking whether entities have already been matched
            tracked_gs_indices = dict()
            for ndx in range(len(gs_entities)):
                tracked_gs_indices[ndx] = False
            tracked_pred_indices = dict()
            for ndx in range(len(pred_entities)):
                tracked_pred_indices[ndx] = False

            # comparison
            for ndx in tracked_gs_indices:
                # do not match against an entity more than once
                if tracked_gs_indices[ndx] is True:
                    continue
                for other_ndx in tracked_pred_indices:
                    # do not match against an entity more than once
                    if tracked_pred_indices[other_ndx] is True:
                        continue

                    # true positives
                    if (not partial and gs_entities[ndx] == pred_entities[other_ndx]) or \
                            (partial and self._intersection(gs_entities[ndx], pred_entities[other_ndx])):
                        tracked_gs_indices[ndx] = True  # mark as matched
                        tracked_pred_indices[other_ndx] = True  # mark as matched
                        gold_standard_arr.append(1)
                        predictions_arr.append(1)
                        break

                # false negatives
                if tracked_gs_indices[ndx] is False:
                    tracked_gs_indices[ndx] = True  # mark as matched
                    gold_standard_arr.append(1)
                    predictions_arr.append(0)

            # false positives
            for _ in range(list(tracked_pred_indices.values()).count(False)):
                gold_standard_arr.append(0)
                predictions_arr.append(1)

        gold_standard_arr = list()
        predictions_arr = list()
        if strictness == "strict":
            for i, gs_entities in enumerate(self._gold_standard):
                pred_entities = self._predictions[i]
                compare_doc_entities()

        elif strictness == "exact":
            for i, gs_entities in enumerate(self._gold_standard):
                # exclude entity types from evaluation
                gs_entities = self._remove_keys(data=gs_entities, keys=["label"])
                pred_entities = self._remove_keys(data=self._predictions[i], keys=["label"])
                compare_doc_entities()

        elif strictness == "type":
            for i, gs_entities in enumerate(self._gold_standard):
                # exclude text spans from evaluation
                gs_entities = self._remove_keys(data=gs_entities, keys=["start", "end"])
                pred_entities = self._remove_keys(data=self._predictions[i], keys=["start", "end"])
                compare_doc_entities()

        elif strictness == "partial":
            for i, gs_entities in enumerate(self._gold_standard):
                # exclude label types from evaluation
                gs_entities = self._remove_keys(data=gs_entities, keys=["label"])
                pred_entities = self._remove_keys(data=self._predictions[i], keys=["label"])
                # represent spans as lists of indices
                gs_entities = [[coord for coord in range(d["start"], d["end"] + 1)] for d in gs_entities]
                pred_entities = [[coord for coord in range(d["start"], d["end"] + 1)] for d in pred_entities]
                compare_doc_entities(partial=True)

        return {
            "precision": round(
                precision_score(y_true=gold_standard_arr, y_pred=predictions_arr, average="micro", labels=[1]) * 100, 2
            ),
            "recall": round(
                recall_score(y_true=gold_standard_arr, y_pred=predictions_arr, average="micro", labels=[1]) * 100, 2
            ),
            "f1": round(
                f1_score(y_true=gold_standard_arr, y_pred=predictions_arr, average="micro", labels=[1]) * 100, 2
            )
        }

    @staticmethod
    def _remove_keys(data: List[dict], keys: list) -> List[dict]:
        """
        delete specified keys from dictionaries in a list of dictionaries
        """
        for d in data:
            for key in keys:
                del d[key]
        return data

    @staticmethod
    def _intersection(list1: list, list2: list) -> bool:
        """
        determines whether 2 lists intersect; return True if yes, otherwise returns False
        """
        return True if len(
            set(list1).intersection(set(list2))
        ) >= 1 else False
