from sklearn.metrics import f1_score

from . import BLURBEvaluator


class GADEvaluator(BLURBEvaluator):
    """
    Implementation of the evaluator for the GAD dataset
    """
    HF_DATASET_PATH = "bigbio/gad"
    HF_DATASET_NAME = "gad_fold0_source"

    def __init__(
            self
    ):
        super(GADEvaluator, self).__init__(
            dataset_name="GAD",
            hf_dataset_path=self.HF_DATASET_PATH,
            hf_dataset_name=self.HF_DATASET_NAME
        )

    def evaluate(self, predictions: dict, split: str = "test") -> dict:
        """
        Evaluate predictions against the GAD gold standard

        Args:
            predictions: dictionary of predictions; must match the relevant provided schema
            split: name of the split to evaluate against

        Returns:
            dictionary containing evaluation metric names as keys and computed evaluation metrics as values
        """
        self.validate_predictions(predictions=predictions)
        gold_standard = self._format_gold_standard(split=split)

        gold_standard_arr = list()
        predictions_arr = list()
        for sentence_id in gold_standard:
            if sentence_id not in predictions["predictions"]:
                raise IOError(f"sentence ID '{sentence_id}' is missing from the predictions")
            gold_standard_arr.append(int(gold_standard[sentence_id]))
            predictions_arr.append(int(predictions["predictions"][sentence_id]))

        output = dict()
        for k, v in predictions.items():
            if k != "predictions":
                output[k] = v
        output["micro_f1"] = round(f1_score(y_true=gold_standard_arr, y_pred=predictions_arr, average="micro") * 100, 2)
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
            key: document ID
            value: dictionary with format:
                - document_id: sentence ID
                - sentence: text of the sentence with @GENE$ and @DISEASE$ mention placeholders
                - label: (deleted if unannotated is True) text of the sentence with @GENE$ and @DISEASE$ mention
                        placeholders
        """
        if split not in self.get_split_names():
            formatted_valid_splits = ', '.join(self.get_split_names())
            raise ValueError(f"expected one of {formatted_valid_splits}; got '{split}'")

        columns_to_remove = ["label"]
        if unannotated is True and as_dataset is True:
            return self._dataset[split].map(lambda example: {}, remove_columns=columns_to_remove)
        if unannotated is True and as_dataset is False:
            return {el["index"]: el for el in
                    self._dataset[split].map(lambda example: {}, remove_columns=columns_to_remove)}
        if unannotated is False and as_dataset is True:
            return self._dataset[split]
        if unannotated is False and as_dataset is False:
            return {el["index"]: el for el in self._dataset[split]}

    def _format_gold_standard(self, split: str) -> dict:
        """
        structure the gold standard annotations to match the prediction schema
        """
        gold_standard = dict()
        for sentence in self._dataset[split]:
            gold_standard[sentence["index"]] = str(sentence["label"])
        return gold_standard
