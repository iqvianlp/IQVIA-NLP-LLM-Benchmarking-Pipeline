# BioNLP Benchmarking
A package for performing BioNLP Benchmarking by evaluating predictions from an NLP system against BLURB datasets

# Sub-packages
## **`blurb`**
### Overview
The `blurb.evaluators` sub-package consists of dataset-specific (or task-specific in the case of the NER datasets) 
classes* that allow for:
1. Load unannotated splits from a dataset
2. Loading the prediction schema for the dataset in question
3. Validating the format of the data structure containing the predictions against the prescribed schema
4. Loading the corresponding gold standard annotations for the dataset split (e.g. train, test)
5. Computing standard evaluation metrics for predictions against the gold standard

*Full list of evaluator classes:
```
BioASQEvaluator()
BIOSSESEvaluator()
ChemProtEvaluator()
DDIEvaluator()
EBMPICOEvaluator()
GADEvaluator()
HoCEvaluator()
PubmedQAEvaluator()
NEREvaluator()
```

### Examples
```python
from tools.bionlp_benchmarking.blurb.evaluators.bioasq import BioASQEvaluator
from utils.io_utils import load_json

evaluator = BioASQEvaluator()
unannotated_data = evaluator.load_split(split="test", unannotated=True, as_dataset=False)
"""
Generate predictions from an NLP system here.
    Format of the predictions dictionary must comply to the schema of the dataset.
"""
predictions = load_json("path_to_predictions_json")
metrics = evaluator.evaluate(predictions, split="test")
```
   