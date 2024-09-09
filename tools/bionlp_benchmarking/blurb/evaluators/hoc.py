from collections import defaultdict

from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from . import BLURBEvaluator


class HoCEvaluator(BLURBEvaluator):
    """
    Implementation of the evaluator for the HoC dataset
    """
    abbr_mapping = {
        0: "GS",
        1: "TPI",
        2: "RI",
        3: "CE",
        4: "CD",
        5: "IM",
        6: "GI",
        8: "A",
        9: "PS",
        10: "ID"
    }
    HF_DATASET_PATH = "bigbio/hallmarks_of_cancer"
    HF_DATASET_NAME = "hallmarks_of_cancer_source"

    def __init__(
            self,
    ):
        super(HoCEvaluator, self).__init__(
            dataset_name="HoC",
            hf_dataset_path=self.HF_DATASET_PATH,
            hf_dataset_name=self.HF_DATASET_NAME
        )

    def evaluate(self, predictions: dict, split: str = "test") -> dict:
        """
        Evaluate predictions against the HoC gold standard

        Args:
            predictions: dictionary of predictions; must match the relevant provided schema
            split: name of the split to evaluate against

        Returns:
            dictionary containing evaluation metric names as keys and computed evaluation metrics as values
        """
        self.validate_predictions(predictions=predictions)
        doc_gold_standard = self._aggregate_preds_by_doc(self._format_gold_standard(split=split))
        doc_preds = self._aggregate_preds_by_doc(predictions["predictions"])

        gold_standard_arr = list()
        predictions_arr = list()

        for doc_id in doc_gold_standard:
            gold_standard_arr.append(doc_gold_standard[doc_id])
            if doc_id in doc_preds:
                predictions_arr.append(doc_preds[doc_id])
            else:
                predictions_arr.append(list())

        gold_standard_arr = MultiLabelBinarizer(
            classes=list(self.abbr_mapping.values())
        ).fit_transform(gold_standard_arr)
        predictions_arr = MultiLabelBinarizer(classes=list(self.abbr_mapping.values())).fit_transform(predictions_arr)

        output = dict()
        for k, v in predictions.items():
            if k != "predictions":
                output[k] = v
        output["average_micro_f1"] = round(
            f1_score(y_pred=gold_standard_arr, y_true=predictions_arr, average="macro") * 100,
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
            key: id of the document
            value: dictionary with format:
                - document_id: id of the document
                - text: text of the document
                - label: (deleted if unannotated is True) label associated to the document
        """
        if split not in self.get_split_names():
            formatted_valid_splits = ', '.join(self.get_split_names())
            raise ValueError(f"expected one of {formatted_valid_splits}; got '{split}'")

        columns_to_remove = ["label"]
        if unannotated is True and as_dataset is True:
            return self._dataset[split].map(lambda example: {}, remove_columns=columns_to_remove)
        if unannotated is True and as_dataset is False:
            return {el["document_id"]: el for el in self._dataset[split].map(lambda example: {}, remove_columns=columns_to_remove)}
        if unannotated is False and as_dataset is True:
            return self._dataset[split]
        if unannotated is False and as_dataset is False:
            return {el["document_id"]: el for el in self._dataset[split]}

    def _format_gold_standard(self, split: str):
        """
        structure the gold standard annotations to match the prediction schema
        """
        gold_standard = dict()
        for sentence in self._dataset[split]:
            gold_standard[sentence["document_id"]] = list()
            for label in sentence["label"]:
                if label == 7:
                    continue
                gold_standard[sentence["document_id"]].append(self.abbr_mapping[label])

        return gold_standard

    @staticmethod
    def _aggregate_preds_by_doc(sent_gold_standard: dict[str, list[str]]) -> dict[str, list[str]]:
        """
        Aggregates sentence-level predictions into document-level predictions.

        Args:
            sent_gold_standard: Sentence-level HoC gold standard.

        Returns:
            dict[str, list[str]]: Document-level gold standard.
        """
        doc_gold_standard = defaultdict(set)
        for sent_id, labels in sent_gold_standard.items():
            doc_id = sent_id.split('_')[0]
            doc_gold_standard[doc_id].update(labels)
        return {doc_id: sorted(labels) for doc_id, labels in doc_gold_standard.items()}
