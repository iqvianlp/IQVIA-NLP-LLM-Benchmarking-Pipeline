# Benchmarking Generative LLMs against BLURB: Part 2/2 - RE, QA, Doc Classification, and Sentence Similarity

These scripts aim at evaluating LLMs against 7 of 13 BLURB datasets, collectively covering the following NLP tasks:
- Relation extraction
- Sentence similarity
- Document classification
- Question answering

## Requirements
- Python 3.9+
- Azure OpenAI Credentials (optional)

### Execution
For in-depth documentation on running the pipeline, run the command:
```shell
python main.py --help
```

## Results
Two sets of results are derived:

- **Scores**: a table of scores is [saved to disk](temp/scores.csv) with following headers:
  - Model: the model name
  - Dataset: the BLURB dataset name
  - Prompt: the name of the prompt template used
  - Score: the overall score of the task, computed by the appropriate [BLURB evaluator class](../../tools/bionlp_benchmarking/blurb/evaluators)
  - Metric: the metric used to derive scores as defined in the evaluator class
- **Predictions**: a table of predictions is [saved to disk](temp/predictions.csv) with the following headers:
  - Model: the model name
  - Dataset: the BLURB dataset name
  - Prompt: the name of the prompt template used
  - Example ID: the ID of the example; the format of which conforms to the example key format defined by the appropriate [BLURB evaluator class](../../tools/bionlp_benchmarking/blurb/evaluators)
  - Prediction (raw): the raw response by the LLM 
  - Valid?: whether the response conforms to the response format defined by the appropriate [BLURB evaluator class](../../tools/bionlp_benchmarking/blurb/evaluators)
  - Prediction (resolved): the value we resolved the raw prediction to, if needed and assuming our resolution strategy is valid (e.g. '4' -> 4.0)
  - Target: The target response, i.e. the 'label(s)' in classification tasks

To customize BLURB tasks and prompt templates, copy `config.yml` and change it as you wish (instructions for changes at
the top of the config file). Then run the experiment pointing at your custom config file.
