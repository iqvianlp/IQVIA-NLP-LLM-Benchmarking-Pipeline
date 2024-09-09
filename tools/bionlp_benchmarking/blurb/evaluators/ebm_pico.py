from sklearn.metrics import f1_score

from . import BLURBEvaluator


class EBMPICOEvaluator(BLURBEvaluator):
    """
    Implementation of the evaluator for the EBM-PICO dataset
    """
    _entity_type_mapping = {"Intervention": "I-INT", "Outcome": "I-OUT", "Participant": "I-PAR"}
    HF_DATASET_PATH = "bigbio/ebm_pico"
    HF_DATASET_NAME = "ebm_pico_source"

    def __init__(
            self,
    ):
        super(EBMPICOEvaluator, self).__init__(
            dataset_name="EBM-PICO",
            hf_dataset_path=self.HF_DATASET_PATH,
            hf_dataset_name=self.HF_DATASET_NAME
        )

    def evaluate(self, predictions: dict, split: str = "test") -> dict:
        """
        Evaluate predictions against the EBM-PICO gold standard

        Args:
            predictions: dictionary of predictions; must match the relevant provided schema
            split: the split of the dataset to evaluate against

        Returns:
            dictionary containing evaluation metric names as keys and computed evaluation metrics as values
        """
        self.validate_predictions(predictions=predictions)
        gold_standard = self._format_gold_standard(split=split)

        gold_standard_arr = list()
        predictions_arr = list()
        for doc_id in gold_standard:
            gold_standard_arr += gold_standard[doc_id]
            if doc_id in predictions["predictions"]:
                if len(predictions["predictions"][doc_id]) != len(gold_standard[doc_id]):
                    raise IOError(
                        f"the number of tokens in 'predictions' for doc ID '{doc_id}', "
                        f"'{len(predictions['predictions'][doc_id])}', does not match the number of tokens in the gold"
                        f" standard, '{len(gold_standard[doc_id])}'"
                    )
                predictions_arr += predictions["predictions"][doc_id]
            else:
                predictions_arr += ["O"] * len(gold_standard[doc_id])

        output = dict()
        for k, v in predictions.items():
            if k != "predictions":
                output[k] = v
        output["macro_f1_word_level"] = round(
            f1_score(
                y_true=gold_standard_arr,
                y_pred=predictions_arr,
                average="macro",
                labels=list(self._entity_type_mapping.values())
            ) * 100,
            2
        )
        return output

    def load_split(self, split: str, unannotated: bool = False, as_dataset: bool = False):
        """
        Load (unannotated) data as a dictionary or as a datasets.Dataset object from a specific split.

        Args:
            split: the dataset split (e.g. train, test) to load
            unannotated: if True, annotations are removed from the split
            as_dataset: if True, the split data are returned as a datasets.Dataset object

        Returns:
            (unannotated) data for the specified split as dictionary with:
            key: ID of the document
            value: dictionary with keys:
                - doc_id: ID of the document
                - text: list of the document
                - entities: (deleted if unannotated is True) list of dictionaries with keys:
                    - text: text of the entity
                    - annotation_type: annotation type of the entity
                    - fine_grained_annotation_type: fin-grained annotation type of the entity
                    - start: start offset of the entity in the text (char based)
                    - end: end offset of the entity in the text (char based)
        """
        if split not in self.get_split_names():
            formatted_valid_splits = ', '.join(self.get_split_names())
            raise(ValueError(f"expected one of {formatted_valid_splits}; got '{split}'"))

        columns_to_remove = ["entities"]
        if unannotated is True and as_dataset is True:
            return self._dataset[split].map(lambda example: {}, remove_columns=columns_to_remove)
        if unannotated is True and as_dataset is False:
            return {el["doc_id"]: el for el in self._dataset[split].map(lambda example: {}, remove_columns=columns_to_remove)}
        if unannotated is False and as_dataset is True:
            return self._dataset[split]
        if unannotated is False and as_dataset is False:
            return {el["doc_id"]: el for el in self._dataset[split]}

    def _format_gold_standard(self, split: str) -> dict:
        """
        structure the gold standard annotations to match the prediction schema
        """
        gold_standard = dict()
        for doc in self._dataset[split]:
            labels = ["O"] * len(doc["text"].strip().split())
            for entity in doc["entities"]:
                start_token_ndx = len(doc["text"][:entity["start"]].split())
                end_token_ndx = start_token_ndx + len(entity["text"].split())
                for ndx in range(start_token_ndx, end_token_ndx):
                    labels[ndx] = self._entity_type_mapping[entity["annotation_type"]]
            gold_standard[doc["doc_id"]] = labels
        return gold_standard
