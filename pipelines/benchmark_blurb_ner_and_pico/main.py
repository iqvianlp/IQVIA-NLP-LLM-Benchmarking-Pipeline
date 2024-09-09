import argparse
import os
import re
from collections import defaultdict
from typing import List

from pathlib import Path
from tqdm import tqdm
import pandas

import tools.llm.deployments as dep
from pipelines.benchmark_blurb_ner_and_pico import (
    FORMATTED_PREDICTIONS_DIR, MARKED_SELECTED_EXAMPLES_DIR, METRICS_DIR, VALID_DATASET_NAMES,
    VALID_NER_DATASET_NAMES, create_model_display_name, document_level_conversion, prompt_llm,
)
from pipelines.benchmark_blurb_other_tasks.constants import HF_MODELS
from tools.bionlp_benchmarking.blurb.evaluators.ebm_pico import EBMPICOEvaluator
from tools.bionlp_benchmarking.blurb.evaluators.ner import NEREvaluator
from tools.llm.clients import AzureClient, TGIClient, VLLMClient
from utils.io_utils import load_json, load_yaml, save_json
from utils.logger import LOGGER


GPT_VERSION_TO_DEPLOYMENT = {
    "3.5": dep.GPT_35_TURBO,
    "4": dep.GPT_4,
    "4_32k": dep.GPT_4_32k
}
CONFIGS_DIR = Path(__file__).parent / "configs"
NER_CONFIG_PATH = CONFIGS_DIR / "ner_config.yml"
CHAT_NER_CONFIG_PATH = CONFIGS_DIR / "chat_ner_config.yml"
PICO_CONFIG_PATH = CONFIGS_DIR / "pico_config.yml"
CHAT_PICO_CONFIG_PATH = CONFIGS_DIR / "chat_pico_config.yml"


def main():
    def command_line():
        parser = argparse.ArgumentParser(
            description="Interface for evaluating LLMs on BLURB NER and PICO datasets."
        )
        subparsers = parser.add_subparsers(help="Steps", dest="step")

        # generate and collect open-source LLM predictions
        prompt_open_source_llm_parser = subparsers.add_parser(
            "prompt-os-llm", help="Generate and collect LLM predictions"
        )
        prompt_open_source_llm_parser.add_argument(
            "--datasets", "-d", nargs="+", type=str, default=VALID_DATASET_NAMES, choices=VALID_DATASET_NAMES,
            help="Space-delimited list of the datasets to process."
        )
        prompt_open_source_llm_parser.add_argument(
            "--model", "-m", type=str, required=True,
            help="Hugging Face model ID of the large language model to prompt."
        )
        prompt_open_source_llm_parser.add_argument(
            "--ip", "-i", type=str, required=True, help="IP address or URL of LLM server."
        )
        prompt_open_source_llm_parser.add_argument(
            "--client", "-cl", type=str, default='tgi', choices=['tgi', 'vllm'],
            help="Type of client to use. Should match the type of server used for the `--ip` parameter."
        )
        prompt_open_source_llm_parser.add_argument(
            '--chat-completion',
            '-cc',
            action='store_true',
            help='Whether to use chat completion prompts when prompting models.'
        )
        prompt_open_source_llm_parser.add_argument(
            '--use-system-role',
            '-usr',
            action='store_true',
            help='Whether to provide initial instructions to the LLM using the "system" role when chat completion is '
                 'enabled.'
        )
        prompt_open_source_llm_parser.add_argument(
            '--random-examples',
            '-re',
            action='store_true',
            help='Whether to use randomly selected examples in few-shot scenarios, instead of similarity matched '
                 'examples'
        )

        # generate and collect OpenAI LLM predictions
        prompt_open_ai_llm_parser = subparsers.add_parser(
            "prompt-oa-llm", help="Generate and collect LLM predictions"
        )
        prompt_open_ai_llm_parser.add_argument(
            "--datasets", "-d", nargs="+", type=str, default=VALID_DATASET_NAMES, choices=VALID_DATASET_NAMES,
            help="Space-delimited list of the datasets to process."
        )
        prompt_open_ai_llm_parser.add_argument(
            "--version", "-v", type=str, default="3.5", choices=list(GPT_VERSION_TO_DEPLOYMENT),
            help="GPT version"
        )
        prompt_open_ai_llm_parser.add_argument(
            '--chat-completion',
            '-cc',
            action='store_true',
            help='Whether to use chat completion prompts when prompting models.'
        )
        prompt_open_ai_llm_parser.add_argument(
            '--use-system-role',
            '-usr',
            action='store_true',
            help='Whether to provide initial instructions to the LLM using the "system" role when chat completion is '
                 'enabled.'
        )
        prompt_open_ai_llm_parser.add_argument(
            '--random-examples',
            '-re',
            action='store_true',
            help='Whether to use randomly selected examples in few-shot scenarios, instead of similarity matched '
                 'examples'
        )

        return parser.parse_args()

    params = command_line()

    if params.step in ["prompt-os-llm", "prompt-oa-llm"]:
        # Determine the correct client to use.
        if params.step == 'prompt-oa-llm':
            deployment = GPT_VERSION_TO_DEPLOYMENT.get(params.version)
            if deployment is None:
                raise ValueError(f"'{params.version}' is not a valid OpenAI deployment")
            client = AzureClient(deployment, params.chat_completion)
        else:
            if params.client == 'vllm':
                client = VLLMClient(params.ip, params.model, chat_completion_enabled=params.chat_completion)
            else:
                client = TGIClient(params.ip, params.model, chat_completion_enabled=params.chat_completion)

        suffix = '_random' if params.random_examples else ''

        # Make predictions.
        for dataset_name in params.datasets:
            examples_filename = os.path.join(
                MARKED_SELECTED_EXAMPLES_DIR,
                f"{dataset_name}_marked_{'random' if params.random_examples else 'selected'}_examples.json")
            if dataset_name in VALID_NER_DATASET_NAMES:
                config = load_yaml(NER_CONFIG_PATH) if not params.chat_completion else \
                    load_yaml(CHAT_NER_CONFIG_PATH)
                for template in config["templates"]:
                    for task in config["tasks"]:
                        if task['dataset'] == dataset_name:
                            LOGGER.info(f"{template['name']} - {task['dataset']}")
                            _ = prompt_llm(
                                template=template,
                                task=task,
                                dataset_name=dataset_name,
                                path_to_examples=examples_filename,
                                client=client,
                                model=params.model if params.step == "prompt-os-llm" else params.version,
                                chat_completion_enabled=params.chat_completion,
                                use_system_role=params.use_system_role,
                                suffix='_random' if params.random_examples else ''
                            )
            else:
                config = load_yaml(PICO_CONFIG_PATH) if not params.chat_completion else \
                    load_yaml(CHAT_PICO_CONFIG_PATH)
                for template in config["templates"]:
                    formatted_output = None
                    for task in config["tasks"]:
                        for entity_type in ['participant', 'intervention', 'outcome']:
                            entity_specific_type = f"{dataset_name}__{entity_type}"
                            if task["dataset"] == entity_specific_type:
                                LOGGER.info(f"{template['name']} - {entity_specific_type}")
                                formatted_output = prompt_llm(
                                    template=template,
                                    task=task,
                                    dataset_name=dataset_name,
                                    path_to_examples=examples_filename,
                                    client=client,
                                    model=params.model if params.step == "prompt-os-llm" else params.version,
                                    formatted_output=formatted_output,
                                    entity_type=entity_type[:3].upper(),
                                    chat_completion_enabled=params.chat_completion,
                                    use_system_role=params.use_system_role,
                                    suffix=suffix
                                )
                    formatted_output = document_level_conversion(formatted_output)
                    formatted_output["dataset_name"] = "EBM PICO"
                    reformatted_template_name = re.sub(r'[,()\- ]', '_', template['name'])
                    model_display_name = create_model_display_name(client.model_name) if params.step == "prompt-os-llm" else\
                        params.version
                    save_json(
                        formatted_output,
                        os.path.join(
                            FORMATTED_PREDICTIONS_DIR,
                            f"{reformatted_template_name}__{dataset_name}"
                            f"{'_openai' if params.step == 'prompt-oa-llm' else ''}_{model_display_name}"
                            f"{'_cc' if params.chat_completion else ''}"
                            f"{'_usr' if params.use_system_role else ''}{suffix}.json"),
                        indent=4
                    )

        # Evaluate.
        LOGGER.info("Evaluating results...")
        evaluate_datasets(params.datasets)
    else:
        raise NotImplementedError("This functionality is not yet implemented.")


def evaluate_datasets(datasets: List[str]):
    predictions_paths = defaultdict(list)
    for dataset_name in datasets:
        for prompt_template in [
            'short__zero_shot__', 'long__zero_shot__', 'short__few_shot__3___', 'long__few_shot__3___'
        ]:
            hf_model_suffixes = [f'_{m.replace("/", "_")}' for m in HF_MODELS]
            model_list = ['_openai_4', '_openai_3.5'] + hf_model_suffixes
            for model in model_list:
                for cc_status in ['_cc', '']:
                    for system_role_status in ['_usr', '']:
                        for selection_type in ['_random', '']:
                            predictions_paths[dataset_name].append(
                                os.path.join(
                                    FORMATTED_PREDICTIONS_DIR,
                                    f"{prompt_template}{dataset_name}{model}{cc_status}{system_role_status}"
                                    f"{selection_type}.json")
                            )
    all_ner_metrics = defaultdict(list)
    all_pico_metrics = defaultdict(list)
    for dataset_name in tqdm(predictions_paths):
        dataset = NEREvaluator(dataset_name=dataset_name) if dataset_name in VALID_NER_DATASET_NAMES else \
            EBMPICOEvaluator()
        for predictions_path in predictions_paths[dataset_name]:
            if not os.path.exists(predictions_path):
                continue
            predictions = load_json(predictions_path)
            _, predictions_filename = os.path.split(predictions_path)
            os.makedirs(METRICS_DIR, exist_ok=True)
            if predictions["model_name"] is None:
                predictions["model_name"] = "openai"
            metrics = dataset.evaluate(predictions=predictions, split="test")
            if dataset_name in VALID_NER_DATASET_NAMES:
                all_ner_metrics["dataset_name"].append(metrics["dataset_name"])
                all_ner_metrics["model_name"].append(metrics["model_name"])
                all_ner_metrics["template"].append(predictions_filename[:-5])
                for s in ("strict", "exact", "partial", "type"):
                    for m in ("precision", "recall", "f1"):
                        all_ner_metrics[f"{s}_{m}"].append(metrics[s][m])
            else:
                all_pico_metrics["dataset_name"].append(metrics["dataset_name"])
                all_pico_metrics["model_name"].append(metrics["model_name"])
                all_pico_metrics["template"].append(predictions_filename[:-5])
                all_pico_metrics["macro_f1_word_level"].append(metrics["macro_f1_word_level"])
    os.makedirs(METRICS_DIR, exist_ok=True)
    pandas.DataFrame(all_ner_metrics, dtype=str).to_csv(
        os.path.join(METRICS_DIR, "ner.csv"), index=False
    )
    pandas.DataFrame(all_pico_metrics, dtype=str).to_csv(
        os.path.join(METRICS_DIR, "pico.csv"), index=False
    )


if __name__ == '__main__':
    main()
