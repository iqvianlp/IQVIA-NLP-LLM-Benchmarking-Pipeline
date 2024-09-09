from sklearn.metrics import accuracy_score

from . import BLURBEvaluator, PROCESSED_DATA_DIR


class BioASQEvaluator(BLURBEvaluator):
    """
    Implementation of the evaluator for the BioASQ dataset


    """

    HF_DATASET_PATH = "bigbio/bioasq_task_b"
    HF_DATASET_NAME = "bioasq_7b_source"
    DATA_DIR = PROCESSED_DATA_DIR

    def __init__(
            self,
    ):
        super(BioASQEvaluator, self).__init__(
            dataset_name="BioASQ",
            hf_dataset_path=self.HF_DATASET_PATH,
            hf_dataset_name=self.HF_DATASET_NAME,
            data_dir=self.DATA_DIR
        )

    def evaluate(self, predictions: dict, split: str = "test") -> dict:
        """
        Evaluate predictions against the BioASQ gold standard

        Args:
            predictions: dictionary of predictions; must match the schema relevant provided schema
            split: name of the split to evaluate against

        Returns:
            dictionary containing evaluation metric names as keys and computed evaluation metrics as values
        """
        self.validate_predictions(predictions=predictions)
        gold_standard = self._format_gold_standard(split=split)

        gold_standard_arr = list()
        predictions_arr = list()
        for question_id in gold_standard:
            if question_id not in predictions["predictions"]:
                raise IOError(f"sentence ID '{question_id}' is missing from the predictions")
            gold_standard_arr.append(gold_standard[question_id])
            predictions_arr.append(predictions["predictions"][question_id])

        output = dict()
        for k, v in predictions.items():
            if k != "predictions":
                output[k] = v
        output["accuracy"] = round(accuracy_score(y_true=gold_standard_arr, y_pred=predictions_arr) * 100, 2)
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
            key: id of the question
            value: dictionary with keys:
                - id: id of the question
                - type: type of the question
                - body: text of the question
                - documents: list of relevant documents (URLs)
                - snippets: list of snippets inside relevant documents, where each snipper is a dict with the following
                            keys:
                            - offsetBeginSection: char offset in begin section of document
                            - offsetEndSection: char offset in end section of document
                            - text: text of the snippet
                            - beginSection: name of begin section of document
                            - endSection: name of end section of document
                            - document: URL of the document
        """
        if split not in self.get_split_names():
            formatted_valid_splits = ', '.join(self.get_split_names())
            raise ValueError(f"expected one of {formatted_valid_splits}; got '{split}'")

        columns_to_remove = ["concepts", "ideal_answer", "exact_answer", "triples"]
        if unannotated is True and as_dataset is True:
            return self._dataset[split].map(lambda example: {}, remove_columns=columns_to_remove)
        if unannotated is True and as_dataset is False:
            return {el["id"]: el for el in self._dataset[split].map(lambda example: {}, remove_columns=columns_to_remove)}
        if unannotated is False and as_dataset is True:
            return self._dataset[split]
        if unannotated is False and as_dataset is False:
            return {el["id"]: el for el in self._dataset[split]}

    def _format_gold_standard(self, split: str):
        """
        structure the gold standard annotations to match the prediction schema
        """
        gold_standard = dict()
        for question in self._dataset[split]:
            if question["type"] == "yesno":
                gold_standard[question["id"]] = question["exact_answer"][0]
        return gold_standard
