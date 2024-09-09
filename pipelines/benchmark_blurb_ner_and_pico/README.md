# Benchmarking Generative LLMs against BLURB: Part 1/2 - NER and PICO

These scripts aim at evaluating LLMs against 6 of 13 BLURB datasets, collectively covering the following NLP tasks:
- Named entity recognition
- PICO (Populations, interventions, comparators, and outcomes)

## Requirements
- Python 3.9+
- Azure OpenAI Credentials (optional)

## Execution
The `main.py` script is configured to support the following workflows:
- `prompt-os-llm` for prompting open-source Hugging Face models.
- `prompt-oa-llm` for prompting Azure OpenAI models.

For high-level documentation of the supported workflows, run the command:
```shell
python main.py --help
```

For in-depth documentation on each workflow, run the command:
```shell
python main.py <workflow> --help
```
