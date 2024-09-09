import os

from sklearn.metrics import f1_score

from . import BLURBEvaluator, PROCESSED_DATA_DIR


class DDIEvaluator(BLURBEvaluator):
    """
    Implementation of the evaluator for the DDI dataset
    """

    def __init__(
            self
    ):
        super(DDIEvaluator, self).__init__(
            dataset_name="DDI",
            json_dataset_path=os.path.join(PROCESSED_DATA_DIR, 'DDI_preprocessed')
        )

    def evaluate(self, predictions: dict, split: str = "test") -> dict:
        """
        Evaluate predictions against the DDI gold standard

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
        for relation_id in gold_standard:
            gold_standard_arr.append(gold_standard[relation_id])
            if relation_id in predictions["predictions"]:
                predictions_arr.append(predictions["predictions"][relation_id])
            else:
                predictions_arr.append("DDI-false")

        output = dict()
        for k, v in predictions.items():
            if k != "predictions":
                output[k] = v

        output["micro_f1"] = round(
            f1_score(
                y_true=gold_standard_arr,
                y_pred=predictions_arr,
                average="micro",
                labels=["DDI-int", "DDI-advise", "DDI-effect", "DDI-mechanism"]
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
            key: document ID
            value: dictionary with format:
                - document_id: document ID
                - text: text of the document
                - entities: list of dictionaries, each one describing a specific entity by means of the following
                            entries:
                            - id: ID of the entity, scoped to the specific document
                            - type: type of the entity
                            - text: text of the entity
                            - offset: list of two integers: start offset and end offset (char base, referring to text)
                - relations: (deleted if unannotated is True) list of dictionaries, each one describing a specific
                            binary relation by means of the following entries:
                            - id: ID of the relation, scoped to the specific document
                            - head: dict with the id of the first entity that participates in that relation ('ref_id'
                                    key) and the role of that entity in the context of the relation ('role' key)
                            - tail: dict with the id of the second entity that participates in that relation ('ref_id'
                                    key) and the role of that entity in the context of the relation ('role' key)
                            - type: list of two integers: start offset and end offset (char base, referring to text)
        """
        if split not in self.get_split_names():
            formatted_valid_splits = ', '.join(self.get_split_names())
            raise(ValueError(f"expected one of {formatted_valid_splits}; got '{split}'"))

        columns_to_remove = ["relations"]
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
        for document in self._dataset[split]:
            for relation in document["relations"]:
                gold_standard[
                    f'{document["document_id"]}_{relation["head"]["ref_id"]}_{relation["tail"]["ref_id"]}'
                ] = f'DDI-{relation["type"].lower()}'
        return gold_standard
