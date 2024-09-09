from sklearn.metrics import accuracy_score

from . import BLURBEvaluator


class PubmedQAEvaluator(BLURBEvaluator):
    """
    Implementation of the evaluator for the PubmedQA dataset
    """
    HF_DATASET_PATH = "bigbio/pubmed_qa"
    HF_DATASET_NAME = "pubmed_qa_labeled_fold0_bigbio_qa"

    def __init__(
            self,
    ):
        super(PubmedQAEvaluator, self).__init__(
            dataset_name="PubmedQA",
            hf_dataset_path=self.HF_DATASET_PATH,
            hf_dataset_name=self.HF_DATASET_NAME
        )

    def evaluate(self, predictions: dict, split: str = "test") -> dict:
        """
        Evaluate predictions against the PubmedQA gold standard

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
                - question_id: id of the question
                - document_id: id of the document
                - type: type of question
                - choices: list with possible answers
                - context: contextual information (string)
                - answer: (deleted if unannotated is True) answer to the question
        """
        if split not in self.get_split_names():
            formatted_valid_splits = ', '.join(self.get_split_names())
            raise ValueError(f"expected one of {formatted_valid_splits}; got '{split}'")

        # Since the test split has distinct columns than the train and validation ones, there are distinct lists of
        # columns to remove in case the parameter unannotated is set equal to True
        columns_to_remove = ["answer"]
        id_column = "question_id"

        if unannotated is True and as_dataset is True:
            return self._dataset[split].map(lambda example: {}, remove_columns=columns_to_remove)
        if unannotated is True and as_dataset is False:
            return {el[id_column]: el for el in self._dataset[split].map(lambda example: {}, remove_columns=columns_to_remove)}
        if unannotated is False and as_dataset is True:
            return self._dataset[split]
        if unannotated is False and as_dataset is False:
            return {el[id_column]: el for el in self._dataset[split]}

    def _format_gold_standard(self, split: str):
        """
        structure the gold standard annotations to match the prediction schema
        """
        gold_standard = dict()
        for question in self._dataset[split]:
            gold_standard[question["question_id"]] = question["answer"]
        return gold_standard
