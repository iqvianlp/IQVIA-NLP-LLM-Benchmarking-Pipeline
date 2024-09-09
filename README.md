# Benchmarking Generative LLMs against BLURB

This repository contains the pipeline code for
[*Evaluation of Large Language Model Performance on the Biomedical Language Understanding and Reasoning Benchmark: Comparative Study*](https://www.medrxiv.org/content/10.1101/2024.05.17.24307411).
It enables the evaluation of generative LLMs on the [Biomedical Language Understanding and Reasoning Benchmark (BLURB)](https://microsoft.github.io/BLURB/),
a collection of biomedical NLP tasks.

The pipeline works Azure OpenAI models or with Hugging Face models served via
[Text Generation Inference](https://github.com/huggingface/text-generation-inference) or [vLLM](https://github.com/vllm-project/vllm).

## Requirements
- Python 3.9+
- Add the top-level directory of this directory to your `PYTHONPATH` environment variable. All script are designed to be run from this directory.
  ```shell
  export PYTHONPATH=$PYTHONPATH:/<path_to_repo>
  ```
- If using Azure OpenAI models, add your credentials to [tools/llm/azure.env](tools/llm/azure.env).
- Hugging Face models must be served via [TGI](https://huggingface.co/docs/text-generation-inference/en/index) or [vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- If using gated models from Hugging Face (such as [Llama 3](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)),
  request access to the gated repository using your Hugging Face account and set the `HF_TOKEN` environment variable:
  ```shell
  export HF_TOKEN=<your_hf_token>
  ```
  

## Installation
- Create and activate a virtual environment
```shell
python -m venv /path/to/venv
source /path/to/venv/bin/activate
```
- Install requirements
```shell
pip install -r requirements.txt
```


## Data
These experiments utilize data from the [BLURB Benchmark](https://microsoft.github.io/BLURB/index.html). To download and
prepare the data, we use Hugging Face Datasets along with modified version of the official BLURB data processing
scripts.

### BioASQ
While most datasets are automatically downloaded, the BioASQ dataset must be downloaded separately by following these
steps:
1. Register and login to the [BioASQ website](http://participants-area.bioasq.org).
2. Download the following files and place them in `/tools/bionlp_benchmarking/blurb/evaluators/resources/processed`:
    - [BioASQ Task 7b training data](http://participants-area.bioasq.org/Tasks/7b/trainingDataset/) (named *BioASQ-training7b.zip*)
    - [BioASQ Task 7b test data](http://participants-area.bioasq.org/Tasks/7b/goldenDataset/) (named *Task7BGoldenEnriched.zip*)

### Data Preparation
Run the following to download and preprocess all data, as well as prepare the example files for few-shot prompting:
```shell
python -X utf8 pipelines/run_data_prep.py
```
**NOTE**: The `-X utf8` option is required for the EBM-PICO dataset to download correctly on Windows systems.


## Benchmarking Pipelines
There are two separate scripts for benchmarking generative LLMs against BLURB: one for the NER and PICO datasets and
one for all other tasks.

Pipeline scripts should be run from root directory of this repository.

See [Reproduction](#reproduction) for the configurations we used in our paper when running each model.

### Common Arguments
There are several common arguments between the two scripts:
- `--chat-completion, -cc`: Uses the chat version of the prompts. Should only be used for chat models. Causes examples for
   few-shot scenarios to be presented as a conversation between the user and the model.
- `--use-system-role, -usr`: Whether to provide initial instructions to the LLM  using the "system" role when chat
   completion is enabled.
- `--ip IP, -i IP`: IP address or URL of LLM server. Not used for OpenAI models, which are configured elsewhere.
- `--client {tgi,vllm}, -cl {tgi,vllm}`: Whether models are hosted with TGI or vLLM. Does not affect OpenAI models.


### NER and PICO Pipeline
#### Usage
To run the pipeline with an Azure OpenAI model, use the following command:
```shell
python pipelines/benchmark_blurb_ner_and_pico/main.py prompt-oa-llm --datasets <LIST-OF-DATASETS> \
    --version <VERSION-OF-OPENAI-MODEL> --chat-completion --use-system-role
```

To run the pipeline with a Hugging Face model hosted on either TGI or vLLM, use the following command:
```shell
python pipelines/benchmark_blurb_ner_and_pico/main.py prompt-os-llm --datasets <LIST-OF-DATASETS> \
    -cl <TGI-OR-VLLM> -i <LLM-SERVER-IP-OR-URL> -m <HUGGINGFACE-MODEL> --chat-completion --use-system-role
```

**Notes:**
- Only use `--chat-completion` and `--use-system-role` for chat models.
- Note that the script **will** overwrite results for previous runs for a given model.
- To run with using random examples instead of semantically similar examples, use the option `--random-examples`.

#### Output
The script will output three directories under `pipelines/benchmark_blurb_ner_and_pico/out`:
- `formatted_predictions`: Processed predictions for each test example.
- `metrics`: CSV files with metrics, one for NER datasets and one for PICO datasets.
- `predictions`: Prompts and raw responses for each test example.

### RE, QA, Doc Classification, and Sentence Similarity Pipeline
#### Usage
To run the pipeline with an Azure OpenAI model or a Hugging Face model, use the following command:
```shell
python pipelines/benchmark_blurb_other_tasks/main.py --datasets <LIST-OF-DATASETS> -cl <TGI-OR-VLLM> \
    -i <LLM-SERVER-IP-OR-URL> -m <HUGGINGFACE-MODEL> --chat-completion --use-system-role
```
Note that several options are not used when using an Azure OpenAI model. See the scripts help message for more details.

**Notes:**
- Only use `--chat-completion` and `--use-system-role` for chat models.
- Note that the script **will not** overwrite results for previous runs for a given model unless the `--overwrite`
  option is used.
- To run with using random examples instead of semantically similar examples, use the option
  `-e pipelines/benchmark_blurb_other_tasks/resources/marked_selected_examples/random_example_map.json`

#### Output
The script will output files and directories under `pipelines/benchmark_blurb_other_tasks/out`:
- `prompts`: A directory containing prompts for each prompting strategy and dataset.
- `predictions.csv`: A CSV containing raw and processed prediction for each test example.
- `prompts.csv`: A CSV containing prompt statistics.
- `scores.csv`: A CSV file containing scores for each prompt-dataset combination.

### Adding More Models
To enable the benchmarking of additional LLMs, add the Hugging Face model ID to the `HF_MODELS` list in
[constants.py](pipelines/benchmark_blurb_other_tasks/constants.py) and set up a TGI or vLLM server serving that model on an accessible endpoint. All generative LLMs
should be compatible.


## Reproduction
To reproduce the experiments as done in the paper, ensure that you use the following options for each model when running
the benchmarking scripts:

**NOTE**:
- `--client` may be either `tgi` or `vllm` except for Flan-T5-XXL, which requires `tgi`.
- See directions above for how to use randomly selected examples instead of semantically similar examples for each
  script.

### OpenAI Models
#### GPT-3.5-Turbo
```shell
python pipelines/benchmark_blurb_ner_and_pico/main.py prompt-oa-llm -v 3.5 -cc -usr -d NCBI-disease BC5-chem BC5-disease JNLPBA BC2GM EBM-PICO
```
```shell
python pipelines/benchmark_blurb_other_tasks/main.py -m gpt-35-turbo-0613 -cc -usr -d BIOSSES ChemProt DDI GAD HoC PubmedQA BioASQ
```

#### GPT-4
```shell
python pipelines/benchmark_blurb_ner_and_pico/main.py prompt-oa-llm -v 4 -cc -usr -d NCBI-disease BC5-chem BC5-disease JNLPBA BC2GM EBM-PICO
```
```shell
python pipelines/benchmark_blurb_other_tasks/main.py -m gpt-4 -cc -usr -d BIOSSES ChemProt DDI GAD HoC PubmedQA BioASQ
```

### Hugging Face Models
#### Flan-T5-XXL
For NER and PICO tasks:
```shell
python pipelines/benchmark_blurb_ner_and_pico/main.py prompt-os-llm -m google/flan-t5-xxl -cl tgi -i <IP-OR-URL> \
     -d NCBI-disease BC5-chem BC5-disease JNLPBA BC2GM EBM-PICO
```
For all other tasks:
```shell
python pipelines/benchmark_blurb_other_tasks/main.py -m google/flan-t5-xxl -cl tgi -i <IP-OR-URL> \
    -d BIOSSES ChemProt DDI GAD HoC PubmedQA BioASQ
```

#### Llama-3-8B-Instruct
For NER and PICO tasks:
```shell
python pipelines/benchmark_blurb_ner_and_pico/main.py prompt-os-llm -m meta-llama/Meta-Llama-3-8B-Instruct -cc -usr \
    -cl vllm -i <IP-OR-URL> -d NCBI-disease BC5-chem BC5-disease JNLPBA BC2GM EBM-PICO
```
For all other tasks:
```shell
python pipelines/benchmark_blurb_other_tasks/main.py -m meta-llama/Meta-Llama-3-8B-Instruct -cc -usr \
    -cl vllm -i <IP-OR-URL> -d BIOSSES ChemProt DDI GAD HoC PubmedQA BioASQ
```

#### Medicine-Llama3-8B
For NER and PICO tasks:
```shell
python pipelines/benchmark_blurb_ner_and_pico/main.py prompt-os-llm -m instruction-pretrain/medicine-Llama3-8B \
    -cl vllm -i <IP-OR-URL> -d NCBI-disease BC5-chem BC5-disease JNLPBA BC2GM EBM-PICO
```
For all other tasks:
```shell
python pipelines/benchmark_blurb_other_tasks/main.py -m instruction-pretrain/medicine-Llama3-8B \
    -cl vllm -i <IP-OR-URL> -d BIOSSES ChemProt DDI GAD HoC PubmedQA BioASQ
```

#### Meditron-7B
For NER and PICO tasks:
```shell
python pipelines/benchmark_blurb_ner_and_pico/main.py prompt-os-llm -m epfl-llm/meditron-7b \
    -cl vllm -i <IP-OR-URL> -d NCBI-disease BC5-chem BC5-disease JNLPBA BC2GM EBM-PICO
```
For all other tasks:
```shell
python pipelines/benchmark_blurb_other_tasks/main.py -m epfl-llm/meditron-7b \
    -cl vllm -i <IP-OR-URL> -d BIOSSES ChemProt DDI GAD HoC PubmedQA BioASQ
```

#### MedLLaMA-13B
For NER and PICO tasks:
```shell
python pipelines/benchmark_blurb_ner_and_pico/main.py prompt-os-llm -m chaoyi-wu/MedLLaMA_13B \
    -cl vllm -i <IP-OR-URL> -d NCBI-disease BC5-chem BC5-disease JNLPBA BC2GM EBM-PICO
```
For all other tasks:
```shell
python pipelines/benchmark_blurb_other_tasks/main.py -m chaoyi-wu/MedLLaMA_13B \
    -cl vllm -i <IP-OR-URL> -d BIOSSES ChemProt DDI GAD HoC PubmedQA BioASQ
```

#### Yi-1.5-34B-Chat
For NER and PICO tasks:
```shell
python pipelines/benchmark_blurb_ner_and_pico/main.py prompt-os-llm -m 01-ai/Yi-1.5-34B-Chat -cc -usr \
    -cl vllm -i <IP-OR-URL> -d NCBI-disease BC5-chem BC5-disease JNLPBA BC2GM EBM-PICO
```
For all other tasks:
```shell
python pipelines/benchmark_blurb_other_tasks/main.py -m 01-ai/Yi-1.5-34B-Chat -cc -usr \
    -cl vllm -i <IP-OR-URL> -d BIOSSES ChemProt DDI GAD HoC PubmedQA BioASQ
```

#### Zephyr-7B-Beta
For NER and PICO tasks:
```shell
python pipelines/benchmark_blurb_ner_and_pico/main.py prompt-os-llm -m HuggingFaceH4/zephyr-7b-beta -cc \
    -cl vllm -i <IP-OR-URL> -d NCBI-disease BC5-chem BC5-disease JNLPBA BC2GM EBM-PICO
```
For all other tasks:
```shell
python pipelines/benchmark_blurb_other_tasks/main.py -m HuggingFaceH4/zephyr-7b-beta -cc \
    -cl vllm -i <IP-OR-URL> -d BIOSSES ChemProt DDI GAD HoC PubmedQA BioASQ
```
