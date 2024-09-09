import os

from sklearn.metrics import f1_score

from . import BLURBEvaluator, PROCESSED_DATA_DIR


class ChemProtEvaluator(BLURBEvaluator):
    """
    Implementation of the evaluator for the ChemProt dataset
    """
    _valid_relations = ["CPR:3", "CPR:4", "CPR:5", "CPR:6", "CPR:9", 'false']

    def __init__(
            self,
    ):
        super(ChemProtEvaluator, self).__init__(
            dataset_name="ChemProt",
            json_dataset_path=os.path.join(PROCESSED_DATA_DIR, 'ChemProt_preprocessed')
        )

    def evaluate(self, predictions: dict, split: str = "test") -> dict:
        """
        Evaluate predictions against the ChemProt gold standard

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
                predictions_arr.append("false")

        output = dict()
        for k, v in predictions.items():
            if k != "predictions":
                output[k] = v
        labels_to_account_for = [r for r in self._valid_relations if 'CPR' in r]
        output["micro_f1"] = round(
            f1_score(
                y_true=gold_standard_arr,
                y_pred=predictions_arr,
                average="micro",
                labels=labels_to_account_for,
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
            key: PubMed ID of the publication
            value: dictionary with format:
                - document_id: ID of the sentence
                - text: text of the publication
                - entities: four lists of the same length identified by the keys 'id', 'type', 'text', 'offset'. At a
                            specific index each list contains the descriptive information of a specific entity. Offsets
                            are tuples of two integers: start offset and end offset (char base, referring to text)
                - relations: (deleted if unannotated is True) three lists of the same length identified by the keys
                            'type', 'arg1', 'arg2'. At a specific index each list contains the descriptive information
                            of a specific relation among a pair of entities: the type of relation ('type'), the
                            identifier of the first entity participating in the relation ('arg1') and the identifier of
                            the second entity participating in the relation ('arg2').
        """
        if split not in self.get_split_names():
            formatted_valid_splits = ', '.join(self.get_split_names())
            raise ValueError(f"expected one of {formatted_valid_splits}; got '{split}'")

        columns_to_remove = ["relations"]
        if unannotated is True and as_dataset is True:
            return self._dataset[split].map(lambda example: {}, remove_columns=columns_to_remove)
        if unannotated is True and as_dataset is False:
            return {
                el["document_id"]: el for el in self._dataset[split].map(
                    lambda example: {}, remove_columns=columns_to_remove
                )
            }
        if unannotated is False and as_dataset is True:
            return self._dataset[split]
        if unannotated is False and as_dataset is False:
            return {el["document_id"]: el for el in self._dataset[split]}

    def _format_gold_standard(self, split: str) -> dict:
        """
        structure the gold standard annotations to match the prediction schema
        """
        gold_standard = dict()
        for doc in self._dataset[split]:
            for ndx, relation in enumerate(doc["relations"]["type"]):
                if relation not in self._valid_relations:
                    continue
                gold_standard[
                    f'{doc["document_id"]}.{doc["relations"]["arg1"][ndx]}.{doc["relations"]["arg2"][ndx]}'
                ] = relation
        return gold_standard
