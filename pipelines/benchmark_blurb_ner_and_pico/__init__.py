import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from nltk.tokenize import wordpunct_tokenize
from tqdm import tqdm

import tools.llm.deployments as dep
from tools.llm.clients import BaseLLMClient, AzureClient
from tools.prompt_gen import get_prompt, get_chat_prompt
from utils.io_utils import load_json, save_json
from utils.logger import LOGGER


VALID_NER_DATASET_NAMES = ["NCBI-disease", "BC5-chem", "BC5-disease", "JNLPBA", "BC2GM"]
VALID_PICO_DATASET_NAMES = ["EBM-PICO"]
VALID_DATASET_NAMES = VALID_NER_DATASET_NAMES + VALID_PICO_DATASET_NAMES
RESOURCES_DIR = Path(__file__).parent / "resources"
OUTPUT_DIR = Path(__file__).parent / "out"
MARKED_SELECTED_EXAMPLES_DIR = RESOURCES_DIR / "marked_selected_examples"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
FORMATTED_PREDICTIONS_DIR = OUTPUT_DIR / "formatted_predictions"
METRICS_DIR = OUTPUT_DIR / "metrics"
TYPE_MAP = {"PAR": "participant", "INT": "intervention", "OUT": "outcome"}


def prompt_llm(
        template: dict,
        task: dict,
        dataset_name: str,
        path_to_examples: str,
        client: BaseLLMClient,
        model: str = None,
        formatted_output: dict = None,
        entity_type: str = None,
        chat_completion_enabled: bool = False,
        use_system_role: bool = False,
        suffix: str = ''
) -> dict[str, Any]:
    """
    Prompts the inputted LLM, collects its predictions, and stores them to disk

    Parameters:
        template: dictionary of task configuration from .yml file containing template-specific prompt specs
        task: dictionary of task configuration from .yml file containing dataset-specific prompt specs
        dataset_name: the dataset to process
        path_to_examples: path to the JSON file containing the pre-selected examples for few-shot workflows
        client: Client to use to prompt the LLM
        model: Hugging Face model ID or OpenAI version
        formatted_output: dictionary of existing predictions to complement.
        entity_type: string denoting the entity; it will be suffixed to all labels of recognized entities,
            B-<entity_type> or I-<entity_type>
        chat_completion_enabled: whether to use the chat completion prompt format to query GPT models.
        use_system_role: whether to provide initial instructions to the LLM using the 'system' role when chat
            completion is enabled.
        suffix: substring to append to the name of the resulting files.

    Returns:
        None
    """
    open_ai = isinstance(client, AzureClient)
    raw_output = defaultdict(lambda: defaultdict(str))
    if formatted_output is None:
        formatted_output = {
            "dataset_name": dataset_name,
            "model_name": model,
            "predictions": dict()
        }
    examples = load_json(path_to_examples)
    for sentence_id in tqdm(list(examples[dataset_name].keys())):
        try:
            prompt = get_prompt(
                task=task,
                template=template,
                inputt=examples[dataset_name][sentence_id]['input'],
                example_map=examples,
                test_example_id=sentence_id,
            ) if not chat_completion_enabled else get_chat_prompt(
                task=task,
                template=template,
                inputt=examples[dataset_name][sentence_id]['input'],
                example_map=examples,
                test_example_id=sentence_id,
                example_output_placeholder='NO_ENTITIES',
                use_system_role=use_system_role
            )
        except IndexError:
            LOGGER.error(f"Index error with {dataset_name} sentence_id: {sentence_id}")
            continue
        try:
            if client.model_name == dep.GPT_4_32k:
                time.sleep(10)
            raw_output[sentence_id]["prediction"] = ""
            if client.is_valid_prompt(prompt):
                raw_output[sentence_id]["prediction"] = client(prompt)
        except Exception as e:
            LOGGER.error(f'Exception encountered: {e} for prompt "{prompt}"')
            raw_output[sentence_id]["prediction"] = "__ERROR__"

        raw_output[sentence_id]["prompt"] = prompt

        formatted_output["predictions"][sentence_id] = postprocess_predictions(
            input_text=examples[dataset_name][sentence_id]['input'][0],
            all_predictions=raw_output[sentence_id]["prediction"].split("|"),
            formatted_prediction=formatted_output["predictions"][
                sentence_id] if sentence_id in formatted_output["predictions"] else None,
            entity_type=entity_type,
            pico=False if dataset_name in VALID_NER_DATASET_NAMES else True
        )

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(FORMATTED_PREDICTIONS_DIR, exist_ok=True)
    reformatted_template_name = re.sub(r'[,()\- ]', '_', template['name'])
    model_display_name = create_model_display_name(model)
    save_json(
        raw_output,
        os.path.join(PREDICTIONS_DIR,
                     f"{reformatted_template_name}__{dataset_name}{'_openai' if open_ai else ''}_"
                     f"{model_display_name}{'_cc' if chat_completion_enabled else ''}"
                     f"{'_usr' if use_system_role else ''}{suffix}.json"),
        indent=4
    )
    save_json(
        formatted_output,
        os.path.join(FORMATTED_PREDICTIONS_DIR,
                     f"{reformatted_template_name}__{dataset_name}{'_openai' if open_ai else ''}_"
                     f"{model_display_name}{'_cc' if chat_completion_enabled else ''}"
                     f"{'_usr' if use_system_role else ''}{suffix}.json"),
        indent=4
    )
    return formatted_output


def postprocess_predictions(
        input_text: str,
        all_predictions: list,
        formatted_prediction: list = None,
        entity_type: str = None,
        pico: bool = False
):
    """
    Formats predictions in a word-level format.

    Parameters:
        input_text: the input text
        all_predictions: list of comma-delimited predicted entities
        formatted_prediction: pre-existing formatted predictions; list of labels
        entity_type: entity type to affix to labels
        pico: whether the text is from a PICO dataset

    Returns:
        list of formatted labels
    """
    if formatted_prediction is None:
        formatted_prediction = len(input_text.split()) * ["O"]
    for predictions in all_predictions:
        predictions = re.sub(r", ", ",", predictions)
        predictions = [p.strip() for p in predictions.split(",")]
        for prediction in predictions:
            prediction = " ".join(wordpunct_tokenize(prediction))
            entity_char_coords = {"start": None, "end": None}
            for _ in range(input_text.count(prediction)):
                entity_char_coords["start"] = input_text.find(
                    prediction, entity_char_coords["end"] if entity_char_coords["end"] is not None else 0
                )
                entity_char_coords["end"] = entity_char_coords["start"] + len(prediction)
                previous_tokens = input_text[: entity_char_coords["start"]].split()
                start_ndx = len(previous_tokens) if entity_char_coords["start"] != 0 else 0
                if previous_tokens and previous_tokens[-1] != input_text.split()[start_ndx - 1]:
                    # Workaround to correct the offset for partial suffix matches at the start of an entity.
                    start_ndx -= 1
                end_ndx = start_ndx + len(prediction.split())
                first_token = True
                try:
                    for ndx in range(start_ndx, end_ndx):
                        if formatted_prediction[ndx] != "O":
                            continue
                        if not pico:
                            formatted_prediction[ndx] = "B" if first_token else "I"
                        else:
                            formatted_prediction[ndx] = "I"
                        if entity_type is not None:
                            formatted_prediction[ndx] += f"-{entity_type}"
                        first_token = False
                except IndexError as e:
                    print("")
                    print(e)
                    print(f"input text: {input_text}")
                    print(f"prediction: {prediction}")
                    print(f'converted prediction text span: {" ".join(input_text.split()[start_ndx: end_ndx])}')

    return formatted_prediction


def document_level_conversion(predictions: dict) -> dict:
    """
    Concatenates sentence-level predictions to document-level.

    Parameters:
        predictions: dictionary of sentence-level predictions

    Returns:
        dictionary of merged predictions
    """
    document_level_predictions = {
        "dataset_name": re.sub("-", " ", predictions["dataset_name"]),
        "model_name": predictions["model_name"],
        "predictions": dict()
    }
    sentence_ids = list(predictions["predictions"].keys())
    sentence_ids.sort()
    for sentence_id in sentence_ids:
        doc_id = sentence_id.split("_")[0]
        if doc_id in document_level_predictions["predictions"]:
            document_level_predictions["predictions"][doc_id] += predictions["predictions"][sentence_id]
        else:
            document_level_predictions["predictions"][doc_id] = predictions["predictions"][sentence_id]

    return document_level_predictions


def create_model_display_name(model_name: str) -> str:
    """
    Normalizes the given model name.

    Args:
        model_name: The Hugging Face model ID or the GPT version.

    Returns:
        str: The normalized model display name.
    """
    return re.sub('/', '_', model_name)
