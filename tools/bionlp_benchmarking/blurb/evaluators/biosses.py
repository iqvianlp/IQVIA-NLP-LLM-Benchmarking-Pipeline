from scipy.stats import pearsonr

from . import BLURBEvaluator


class BIOSSESEvaluator(BLURBEvaluator):
    """
    Implementation of the evaluator for the BIOSSES dataset
    """
    HF_DATASET_PATH = "bigbio/biosses"
    HF_DATASET_NAME = "biosses_source"

    def __init__(
            self,
    ):
        super(BIOSSESEvaluator, self).__init__(
            dataset_name="BIOSSES",
            hf_dataset_path=self.HF_DATASET_PATH,
            hf_dataset_name=self.HF_DATASET_NAME
        )

    def evaluate(self, predictions: dict, split: str = "test") -> dict:
        """
        Evaluate predictions against the BIOSSES gold standard

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
        for doc_id in gold_standard:
            if doc_id not in predictions["predictions"]:
                raise IOError(f"document ID '{doc_id}' is missing from the predictions")
            gold_standard_arr.append(gold_standard[doc_id])
            predictions_arr.append(predictions["predictions"][doc_id])

        output = dict()
        for k, v in predictions.items():
            if k != "predictions":
                output[k] = v
        output["pearson"] = round(pearsonr(gold_standard_arr, predictions_arr).statistic, 4)

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
            key: id of the sample
            value: dictionary with format:
                - id: id of the sample
                - document_id: document ID
                - text_1: first text
                - text_2: second text
                - annotator_a: (deleted if unannotated is True) evaluation of text similarity provided by annotator a
                - annotator_b: (deleted if unannotated is True) evaluation of text similarity provided by annotator b
                - annotator_c: (deleted if unannotated is True) evaluation of text similarity provided by annotator c
                - annotator_d: (deleted if unannotated is True) evaluation of text similarity provided by annotator d
                - annotator_e: (deleted if unannotated is True) evaluation of text similarity provided by annotator e
        """
        if split not in self.get_split_names():
            formatted_valid_splits = ', '.join(self.get_split_names())
            raise ValueError(f"expected one of {formatted_valid_splits}; got '{split}'")

        columns_to_remove = ["annotator_a", "annotator_b", "annotator_c", "annotator_d", "annotator_e"]
        if unannotated is True and as_dataset is True:
            return self._dataset[split].map(lambda example: {}, remove_columns=columns_to_remove)
        if unannotated is True and as_dataset is False:
            return {el["id"]: el for el in self._dataset[split].map(lambda example: {}, remove_columns=columns_to_remove)}
        if unannotated is False and as_dataset is True:
            return self._dataset[split]
        if unannotated is False and as_dataset is False:
            return {el["id"]: el for el in self._dataset[split]}

    def _format_gold_standard(self, split: str) -> dict:
        """
        structure the gold standard annotations to match the prediction schema
        """
        gold_standard = dict()
        annotators = ["a", "b", "c", "d", "e"]
        for doc in self._dataset[split]:
            gold_standard[str(doc["document_id"])] = sum([doc[f"annotator_{i}"] for i in annotators])/len(annotators)
        return gold_standard
