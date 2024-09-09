from copy import deepcopy

from . import BLURBEvaluator, SCHEMAS_DIR, NERMetricComputation


class NEREvaluator(BLURBEvaluator):
    """
    Implementation of the evaluator for the BC2GM dataset
    """

    DATASET_TO_HF_NAME = {
        'BC2GM': 'bc2gm',
        'BC5-chem': 'bc5chem',
        'BC5-disease': 'bc5disease',
        'JNLPBA': 'jnlpba',
        'NCBI-disease': 'ncbi_disease'
    }

    def __init__(
            self,
            dataset_name: str,
    ):
        schema_path = f"{SCHEMAS_DIR}/named_entity_recognition.json"
        super(NEREvaluator, self).__init__(
            dataset_name=dataset_name,
            schema_path=schema_path,
            hf_dataset_path="bigbio/blurb",
            hf_dataset_name=self.DATASET_TO_HF_NAME[dataset_name]
        )

    def evaluate(self, predictions: dict, split: str = "test", tag_type: str = "UKN") -> dict:
        """
        Evaluate predictions against the BC2GM gold standard

        Args:
            predictions: dictionary of predictions; must match the relevant provided schema
            split: split of the dataset to evaluate against
            tag_type: the entity type

        Returns:
            dictionary containing evaluation metric names as keys and computed evaluation metrics as values
        """
        self.validate_predictions(predictions=predictions)
        gold_standard_entities, predictions_entities = self._format_data(
            gold_standard_split=split,
            predictions=predictions["predictions"],
            tag_type=tag_type
        )
        output = dict()
        for k, v in predictions.items():
            if k != "predictions":
                output[k] = v
        for strictness in ["strict", "exact", "partial", "type"]:
            output[strictness] = NERMetricComputation(
                deepcopy(gold_standard_entities), deepcopy(predictions_entities)
            ).compute(strictness=strictness)
        return output

    def load_split(self, split: str, unannotated: bool = False, as_dataset: bool = False) -> dict:
        """
        Load (unannotated) data as a dictionary or as a datasets.Dataset object from a specific split.

        Args:
            split: the dataset split (e.g. train, test) to load
            unannotated: if True, annotations are removed from the split
            as_dataset: if True, the split data are returned as a datasets.Dataset object

        Returns:
            (unannotated) data for the specified split as dictionary with:
            key: ID of the text
            value: dictionary with keys:
                - id: ID of the text
                - tokens: list of tokens of the text
                - type: (deleted if unannotated is True) type of entity
                - ner_tags: list of NER tags (0 = out, 1 = beginning, 2 = inside)
        """
        if split not in self.get_split_names():
            formatted_valid_splits = ', '.join(self.get_split_names())
            raise(ValueError(f"expected one of {formatted_valid_splits}; got '{split}'"))

        columns_to_remove = ["type", "ner_tags"]
        if unannotated is True and as_dataset is True:
            return self._dataset[split].map(lambda example: {}, remove_columns=columns_to_remove)
        if unannotated is True and as_dataset is False:
            return {el["id"]: el for el in self._dataset[split].map(lambda example: {}, remove_columns=columns_to_remove)}
        if unannotated is False and as_dataset is True:
            return self._dataset[split]
        if unannotated is False and as_dataset is False:
            return {el["id"]: el for el in self._dataset[split]}

    def _format_data(self, gold_standard_split: str, predictions: dict, tag_type: str):
        """
        structure the gold standard and predicted annotations to match the NERMetricComputation class format
        requirements
        """
        gold_standard_arr = list()
        predictions_arr = list()
        for document in self._dataset[gold_standard_split]:
            gold_standard_arr.append(list())
            for ndx, numeric_tag in enumerate(document["ner_tags"]):
                if numeric_tag == 1:
                    gold_standard_arr[-1].append({"label": tag_type, "start": ndx, "end": ndx})
                elif numeric_tag == 2:
                    gold_standard_arr[-1][-1]["end"] = ndx

            predictions_arr.append(list())
            if document["id"] in predictions:
                for ndx, tag in enumerate(predictions[document["id"]]):
                    if tag == "B":
                        predictions_arr[-1].append({"label": tag_type, "start": ndx, "end": ndx})
                    elif tag == "I":
                        predictions_arr[-1][-1]["end"] = ndx
        return gold_standard_arr, predictions_arr
