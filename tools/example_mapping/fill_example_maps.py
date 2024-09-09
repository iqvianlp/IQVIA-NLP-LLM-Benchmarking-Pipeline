import argparse
import statistics
from copy import deepcopy
from typing import Any, Dict, Tuple, Union

from tqdm import tqdm

from pipelines.benchmark_blurb_ner_and_pico import MARKED_SELECTED_EXAMPLES_DIR
from pipelines.benchmark_blurb_other_tasks.constants import *
from pipelines.benchmark_blurb_other_tasks.common import augment_entry, load_hoc_maps, normalize_type
from tools.bionlp_benchmarking.blurb.evaluators.bioasq import BioASQEvaluator
from tools.bionlp_benchmarking.blurb.evaluators.biosses import BIOSSESEvaluator
from tools.bionlp_benchmarking.blurb.evaluators.chemprot import ChemProtEvaluator
from tools.bionlp_benchmarking.blurb.evaluators.ddi import DDIEvaluator
from tools.bionlp_benchmarking.blurb.evaluators.ebm_pico import EBMPICOEvaluator
from tools.bionlp_benchmarking.blurb.evaluators.gad import GADEvaluator
from tools.bionlp_benchmarking.blurb.evaluators.hoc import HoCEvaluator
from tools.bionlp_benchmarking.blurb.evaluators.ner import NEREvaluator
from tools.bionlp_benchmarking.blurb.evaluators.pubmedqa import PubmedQAEvaluator
from utils.io_utils import load_json, save_json
from utils.logger import LOGGER

# Default directories
PARENT_DIR = Path(__file__).parent
NER_AND_PICO_MAP_DIR = PARENT_DIR / 'resources' / 'ner_and_pico_maps'
OTHER_TASK_MAP_DIR = PARENT_DIR / 'resources' / 'other_task_maps'
OTHER_TASK_OUTPUT_DIR = EXAMPLE_MAP_PATH.parent


HOC_LABEL_TO_ABBR_MAP, HOC_ABBR_TO_LABEL_MAP = load_hoc_maps()
DATASET_TO_EVALUATOR_CLASS = {
    'BIOSSES': BIOSSESEvaluator,
    'ChemProt': ChemProtEvaluator,
    'DDI': DDIEvaluator,
    'GAD': GADEvaluator,
    'HoC': HoCEvaluator,
    'PubmedQA': PubmedQAEvaluator,
    'BioASQ': BioASQEvaluator
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fills in example maps used for generating prompts with input text and output labels.'
    )
    parser.add_argument(
        '--ner_and_pico_example_map_dir',
        '-nm',
        required=False,
        type=Path,
        default=NER_AND_PICO_MAP_DIR,
        help='Directory containing NER and PICO example maps in JSON format.'
    )
    parser.add_argument(
        '--other_task_example_map_dir',
        '-om',
        required=False,
        type=Path,
        default=OTHER_TASK_MAP_DIR,
        help='Directory containing example maps for non-NER/PICO tasks in JSON format.'
    )
    parser.add_argument(
        '--ner_and_pico_out_dir',
        '-no',
        required=False,
        type=Path,
        default=MARKED_SELECTED_EXAMPLES_DIR,
        help='Directory to output filled-in examples for NER and PICO datasets.'
    )
    parser.add_argument(
        '--other_task_out_dir',
        '-oo',
        required=False,
        type=Path,
        default=OTHER_TASK_OUTPUT_DIR,
        help='Directory to output filled-in examples for non-NER/PICO datasets.'
    )
    return parser.parse_args()


def get_data_keys(dataset_name: str) -> Tuple[str, Union[str, None], str]:
    """
    Returns the keys to the input field(s) and the target field for the given dataset.
    """
    second_input_key = None
    if dataset_name == BIOSSES:
        input_key = 'text_1'
        second_input_key = 'text_2'
        output_key = [f'annotator_{x}' for x in ['a', 'b', 'c', 'd', 'e']]
    elif dataset_name == CHEMPROT:
        input_key = 'text'
        output_key = 'relations'
    elif dataset_name == DDI:
        input_key = 'text'
        output_key = 'relations'
    elif dataset_name == GAD:
        input_key = 'sentence'
        output_key = 'label'
    elif dataset_name == HOC:
        input_key = 'text'
        output_key = 'label'
    elif dataset_name == PUBMEDQA:
        input_key = 'question'
        second_input_key = 'context'
        output_key = 'answer'
    else:  # BioASQ; NB: dataset names are validated at the start of the loop, so it must be of one these at this stage
        input_key = 'body'
        output_key = 'exact_answer'  # there's "ideal_answer" too, but they're long; prompt says "yes" or "no" only
    return input_key, second_input_key, output_key


def _get_example_texts(dataset_name: str, train_example: Dict[str, Any], input_key: str, output_key: str):
    input_text = train_example[input_key]
    if dataset_name == BIOSSES:  # return the average of answers by 5 annotators (how the evaluator class does it)
        candidates = [train_example[x] for x in output_key]
        output_text = statistics.fmean(candidates)
    elif dataset_name == CHEMPROT:
        current_example = deepcopy(train_example)
        relations = current_example[output_key]
        # If relations exist, pick entities of first relation:
        if relations['arg1'] and relations['arg2']:
            entity_1 = relations['arg1'][0]
            entity_2 = relations['arg2'][0]
        # If no relations, pick first two entities of different types, or just the first two if all are the same
        # type; if only one entity (there's always at least 1 entity!), just repeat them:
        else:
            entity_1 = current_example['entities']['id'][0]
            entity_1_type = normalize_type(current_example['entities']['type'][0], dataset_name == DDI)
            if len(current_example['entities']['id']) > 1:
                entity_2 = current_example['entities']['id'][1]
                for idx, t in enumerate(current_example['entities']['type']):
                    if normalize_type(t, dataset_name == DDI) != entity_1_type:
                        entity_2 = current_example['entities']['id'][idx]
                        break
            else:
                entity_2 = entity_1
        augment_entry(current_example, (entity_1, entity_2))
        input_text = current_example['text_annotated']
        # Return output for first relation in example only; if none or not a valid relation, return "false":
        if relations['type'] and relations['type'][0] in ChemProtEvaluator._valid_relations:
            output_text = relations['type'][0]
        else:
            output_text = 'false'
    elif dataset_name == DDI:
        current_example = deepcopy(train_example)
        relations = current_example[output_key]
        # Return input/output for first relation in example only; if none, return unannotated text and "DDI-false":
        if relations:
            entity_1 = relations[0]['head']['ref_id']
            entity_2 = relations[0]['tail']['ref_id']
            output_label = relations[0]['type']
        else:  # DDI data is filtered upstream so that it always has 2+ drugs
            entity_1 = current_example['entities'][0]['id']
            entity_2 = current_example['entities'][1]['id']
            output_label = 'false'
        augment_entry(current_example, (entity_1, entity_2), is_ddi=True)
        input_text = current_example['text_annotated']
        output_text = f'DDI-{output_label.lower()}'
    elif dataset_name == HOC:
        result_parts = []
        for idx in sorted(train_example[output_key]):
            if idx != 7:
                result_parts.append(HOC_ABBR_TO_LABEL_MAP[HoCEvaluator.abbr_mapping[idx]])
        output_text = ', '.join(result_parts)
    elif dataset_name == PUBMEDQA or dataset_name == BIOASQ:
        answers = train_example[output_key]
        if len(answers) > 1:
            LOGGER.warning(f'{dataset_name} example "{train_example["id"]}" has more than one answer: '
                           f'{", ".join(answers)}')
        output_text = answers[0]
    else:  # GAD
        output_text = train_example[output_key]

    return input_text, output_text


def format_ner_examples_for_dataset(
        train_set: dict,
        test_set: dict,
        dataset_name: str,
        annotations: dict,
        path_selected_examples_json: str
) -> dict:
    """
    Combines example outputs and their corresponding input text in a format compatible with the prompt generator

    Args:
        train_set: the dataset's train split
        test_set: the dataset's test split
        dataset_name: name of the dataset
        annotations: formatted annotated text (output example)
        path_selected_examples_json: path to JSON file containing the selected examples

    Returns:
        None
    """
    unfilled_selected_examples = load_json(path_selected_examples_json)
    selected_examples = unfilled_selected_examples[dataset_name]
    marked_selected_examples = {dataset_name: dict()}

    for input_id in tqdm(selected_examples):
        marked_selected_examples[dataset_name][input_id] = {
            "input": [' '.join(test_set[input_id]["tokens"])]
        }
        marked_selected_examples[dataset_name][input_id]["sims"] = list()
        for example_type in selected_examples[input_id]["examples"]:
            for ex_id in selected_examples[input_id]["examples"][example_type]["id"]:
                marked_selected_examples[dataset_name][input_id]["sims"].append(
                    {
                        "input": [
                            ' '.join(train_set[ex_id]['tokens'])
                        ],
                        "output": annotations[ex_id]
                    }
                )
    return marked_selected_examples


def format_pico_selected_examples_for_comma_delimited_prompting(
        dataset_name: str, path_selected_examples_json: Union[str, Path]
):
    """
    Formats selected examples for prompt generator and stores them to disk

    Parameters:
        dataset_name: name of an NER dataset
        path_selected_examples_json: the path to JSON files containing the selected examples

    Returns:
        None
    """
    evaluator = EBMPICOEvaluator()
    test_set = evaluator.load_split('test')
    selected_examples = load_json(path_selected_examples_json)
    marked_selected_examples = {dataset_name: dict()}
    for input_id in tqdm(selected_examples[dataset_name]):
        marked_selected_examples[dataset_name][input_id] = {
            "input": [test_set[input_id]["text"]]
        }
    return marked_selected_examples


def format_ner_selected_examples_for_comma_delimited_prompting(
        dataset_name: str, path_selected_examples_json: Union[str, Path]
) -> dict:
    """
    Formats selected examples for prompt generator and stores them to disk

    Args:
        dataset_name: name of an NER dataset
        path_selected_examples_json: path to JSON file containing the selected examples

    Returns:
        None
    """
    evaluator = NEREvaluator(dataset_name)
    train_set = evaluator.load_split('train')
    test_set = evaluator.load_split('test')
    annotations = {}
    examples = list(train_set.values())
    for ndx, text in enumerate(tqdm([ex['tokens'] for ex in examples])):
        annotated_entities = list()
        for word_ndx, label in enumerate(examples[ndx]['ner_tags']):
            if label == 1:
                annotated_entities.append([text[word_ndx]])
            elif label == 2:
                annotated_entities[-1].append(text[word_ndx])
        annotations[examples[ndx]["id"]] = ", ".join([" ".join(i) for i in annotated_entities])

    return format_ner_examples_for_dataset(
        train_set=train_set,
        test_set=test_set,
        dataset_name=dataset_name,
        annotations=annotations,
        path_selected_examples_json=path_selected_examples_json
    )


def fill_other_task_example_maps(
        selected_examples_dir: Union[str, Path],
        out_dir: Union[str, Path],
) -> None:
    """
    Formats selected examples for prompt generator for non-NER and PICO tasks and stores them to disk.

    Args:
        selected_examples_dir: path to directory containing JSON files of the selected examples
        out_dir: path directory to write JSON files with examples filled in

    Returns:
        None
    """
    selected_examples_dir = Path(selected_examples_dir)
    for example_map_path in selected_examples_dir.glob('*.json'):
        selected_examples = load_json(example_map_path)
        marked_selected_examples = {}
        for dataset_name, examples_dict in selected_examples.items():
            evaluator = DATASET_TO_EVALUATOR_CLASS[dataset_name]()
            train_set = evaluator.load_split('train')
            test_set = evaluator.load_split('test')
            marked_selected_examples[dataset_name] = {}
            input_key, second_input_key, output_key = get_data_keys(dataset_name)
            for test_id, examples in tqdm(examples_dict.items()):
                # Get input example
                try:
                    test_ex = test_set[test_id]
                except KeyError:
                    test_ex = test_set[int(test_id)]
                ex_text = [test_ex[input_key]]
                if second_input_key is not None:
                    ex_text.append(test_ex[second_input_key])
                marked_selected_examples[dataset_name][str(test_id)] = {
                    'input': ex_text,
                    'sims': []
                }
                # Get few-shot examples
                for ex_id in examples['examples']['positive']['id']:
                    train_ex = train_set[ex_id]
                    input_text, output_text = _get_example_texts(dataset_name, train_ex, input_key, output_key)
                    sim_dict = {
                        'input': [input_text],
                        'output': output_text
                    }
                    if second_input_key is not None:
                        sim_dict['input'].append(train_ex[second_input_key])
                    marked_selected_examples[dataset_name][str(test_id)]['sims'].append(sim_dict)
        filename_map = {
            'selected_examples.json': 'example_map.json',
            'random_examples.json': 'random_example_map.json'
        }
        out_filename = filename_map[example_map_path.name]
        save_json(marked_selected_examples, out_dir / out_filename, indent=2)


def fill_ner_and_pico_example_maps(
        selected_examples_dir: Union[str, Path],
        out_dir: Union[str, Path]
) -> None:
    """
    Formats selected examples for prompt generator for NER and PICO datasets and stores them to disk.

    Args:
        selected_examples_dir: path to directory containing JSON files of the selected examples
        out_dir: directory to write JSON files with examples filled in

    Returns:
        None
    """
    selected_examples_dir = Path(selected_examples_dir)
    for example_map_path in selected_examples_dir.glob('*.json'):
        dataset_name, rest = example_map_path.stem.split('_', maxsplit=1)
        out_filename = f'{dataset_name}_marked_{rest}.json'
        if dataset_name in ['BC2GM', 'BC5-chem', 'BC5-disease', 'JNLPBA', 'NCBI-disease']:
            formatted_examples = format_ner_selected_examples_for_comma_delimited_prompting(
                dataset_name,
                example_map_path
            )
        else:  # if EMB-PICO
            formatted_examples = format_pico_selected_examples_for_comma_delimited_prompting(
                dataset_name,
                example_map_path
            )
        save_json(formatted_examples, out_dir / out_filename, indent=4)


def fill_examples_for_all_datasets(
        ner_and_pico_example_map_dir: Path,
        ner_and_pico_out_dir: Path,
        other_task_example_map_dir: Path,
        other_task_out_dir: Path
) -> None:
    """

    Args:
        ner_and_pico_example_map_dir: Directory containing JSON files of selected NER and PICO examples
        ner_and_pico_out_dir: Directory to write JSON files with NER and PICO examples filled in
        other_task_example_map_dir: Directory containing JSON files of the selected examples for other tasks
        other_task_out_dir: Directory to write JSON files with examples for other tasks filled in

    Returns:
        None
    """
    ner_and_pico_out_dir.mkdir(parents=True, exist_ok=True)
    other_task_out_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info('Filling in examples for NER and PICO tasks...')
    fill_ner_and_pico_example_maps(
        ner_and_pico_example_map_dir,
        ner_and_pico_out_dir
    )
    LOGGER.info('Filling in examples for other tasks...')
    fill_other_task_example_maps(
        other_task_example_map_dir,
        other_task_out_dir
    )


def main() -> None:
    args = parse_args()
    fill_examples_for_all_datasets(
        args.ner_and_pico_example_map_dir,
        args.ner_and_pico_out_dir,
        args.other_task_example_map_dir,
        args.other_task_out_dir
    )


if __name__ == '__main__':
    main()
