from collections import defaultdict
import re
from pathlib import Path

from pipelines.benchmark_blurb_other_tasks.constants import *
from utils.io_utils import load_json
from tools.bionlp_benchmarking.blurb.evaluators.bioasq import BioASQEvaluator
from tools.bionlp_benchmarking.blurb.evaluators.biosses import BIOSSESEvaluator
from tools.bionlp_benchmarking.blurb.evaluators.chemprot import ChemProtEvaluator
from tools.bionlp_benchmarking.blurb.evaluators.ddi import DDIEvaluator
from tools.bionlp_benchmarking.blurb.evaluators.gad import GADEvaluator
from tools.bionlp_benchmarking.blurb.evaluators.hoc import HoCEvaluator
from tools.bionlp_benchmarking.blurb.evaluators.pubmedqa import PubmedQAEvaluator


def load_hoc_maps():
    return load_json(HOC_LABEL_TO_ABBR_MAP_FILE), load_json(HOC_ABBR_TO_LABEL_MAP_FILE)


def hoc_ints_to_str(ints):
    return [HoCEvaluator.abbr_mapping[x] for x in ints]


def load_evaluator(dataset_name):
    if dataset_name == BIOSSES:
        return BIOSSESEvaluator()
    elif dataset_name == CHEMPROT:
        return ChemProtEvaluator()
    elif dataset_name == DDI:
        return DDIEvaluator()
    elif dataset_name == GAD:
        return GADEvaluator()
    elif dataset_name == HOC:
        return HoCEvaluator()
    elif dataset_name == PUBMEDQA:
        return PubmedQAEvaluator()
    else:  # BioASQ; NB: dataset names are validated at the start of the loop, so it must be one of these at this stage
        return BioASQEvaluator()


def filter_data(logger, dataset_name, dataset, limit=None, is_train=False):
    """Filters datat sets (train or test split) so that they make sense for prompts.

    Constraints:
        - PubmedQA: only yes/no/maybe questions
        - BioASQ: only yes/no questions
        - DDI: only examples with 2+ entities
    """
    data_type = 'train' if is_train else 'test'

    data_size_before = len(dataset)
    if dataset_name == PUBMEDQA:
        dataset = {k: v for k, v in dataset.items() if set(v['choices']) == {'yes', 'no', 'maybe'}}
    elif dataset_name == BIOASQ:
        dataset = {k: v for k, v in dataset.items() if v['type'] == 'yesno'}
    elif dataset_name == DDI:
        dataset = {k: v for k, v in dataset.items() if len(v['entities']) > 1}

    data_size_after = len(dataset)
    if data_size_after < data_size_before:
        diff = data_size_before - data_size_after
        logger.info(f'{diff} {data_type} examples filtered out with invalid format for prompt')

    if limit:
        limit = limit * 3 if is_train else limit
        logger.info(f'Limiting remaining {data_type} data to {limit} example(s)')
        dataset = {k: dataset[k] for k in list(dataset.keys())[:limit]}

    return dataset


def augment_entry(entry, pair, is_ddi=False):
    if is_ddi:
        id_key = 'document_id'
        sep = '_'
        a, b = None, None
        for e in entry['entities']:
            if e['id'] == pair[0]:
                a = e
            elif e['id'] == pair[1]:
                b = e
            if a is not None and b is not None:
                break
        a_type = a['type']
        b_type = b['type']
        insertions = {
            'offsets': [
                a['offsets'],
                b['offsets']
            ],
            'type': [a_type, b_type]
        }
    else:
        id_key = 'document_id'
        sep = '.'
        a_idx = entry['entities']['id'].index(pair[0])
        b_idx = entry['entities']['id'].index(pair[1])
        a_type = entry['entities']['type'][a_idx]
        b_type = entry['entities']['type'][b_idx]
        insertions = {
            'offsets': [
                entry['entities']['offsets'][a_idx],
                entry['entities']['offsets'][b_idx]
            ],
            'type': [a_type, b_type]
        }
    entry['example_id'] = f'{entry[id_key]}{sep}{pair[0]}{sep}{pair[1]}'
    entry['text_annotated'] = _get_annotated_text(insertions, entry['text'], is_ddi=is_ddi)
    return entry


def _get_annotated_text(insertions, raw_input, is_ddi=False):
    # The entities share at least one start or end coordinate
    if (insertions['offsets'][0][0] == insertions['offsets'][1][0]) or (
            insertions['offsets'][0][1] == insertions['offsets'][1][1]):
        # Entities share the same start coordinate and the same end coordinate
        if (insertions['offsets'][0][0] == insertions['offsets'][1][0]) and \
                (insertions['offsets'][0][1] == insertions['offsets'][1][1]):
            ordered_coordinates = [
                (insertions['offsets'][0][0], insertions['type'][0]),
                (insertions['offsets'][1][0], insertions['type'][1]),
                (insertions['offsets'][1][1], insertions['type'][1]),
                (insertions['offsets'][0][1], insertions['type'][0])
            ]
        # Entities share the same start coordinate
        elif insertions['offsets'][0][0] == insertions['offsets'][1][0]:
            # End of entity 1 comes first
            if insertions['offsets'][0][1] < insertions['offsets'][1][1]:
                ordered_coordinates = [
                    (insertions['offsets'][1][0], insertions['type'][1]),
                    (insertions['offsets'][0][0], insertions['type'][0]),
                    (insertions['offsets'][0][1], insertions['type'][0]),
                    (insertions['offsets'][1][1], insertions['type'][1])
                ]
            # End of entity 2 comes first
            else:
                ordered_coordinates = [
                    (insertions['offsets'][0][0], insertions['type'][0]),
                    (insertions['offsets'][1][0], insertions['type'][1]),
                    (insertions['offsets'][1][1], insertions['type'][1]),
                    (insertions['offsets'][0][1], insertions['type'][0])
                ]
        # Entities share the same end coordinate
        elif insertions['offsets'][0][1] == insertions['offsets'][1][1]:
            # Start of entity 1 comes first
            if insertions['offsets'][0][0] < insertions['offsets'][1][0]:
                ordered_coordinates = [
                    (insertions['offsets'][0][0], insertions['type'][0]),
                    (insertions['offsets'][1][0], insertions['type'][1]),
                    (insertions['offsets'][1][1], insertions['type'][1]),
                    (insertions['offsets'][0][1], insertions['type'][0])
                ]
            # Start of entity 2 comes first
            else:
                ordered_coordinates = [
                    (insertions['offsets'][1][0], insertions['type'][1]),
                    (insertions['offsets'][0][0], insertions['type'][0]),
                    (insertions['offsets'][0][1], insertions['type'][0]),
                    (insertions['offsets'][1][1], insertions['type'][1])
                ]

    # The entities do not share a start or end coordinate
    else:
        coord_to_type = defaultdict(list)
        for ndx, type in enumerate(insertions['type']):
            coord_to_type[insertions['offsets'][ndx][0]].append(type)
            coord_to_type[insertions['offsets'][ndx][1]].append(type)
        ordered_coordinates_set = list(coord_to_type.keys())
        ordered_coordinates_set.sort()
        ordered_coordinates = list()
        for k in ordered_coordinates_set:
            ordered_coordinates += [(k, t) for t in coord_to_type[k]]

    output = ''
    chunk_start = 0
    for coord in ordered_coordinates:
        output += raw_input[chunk_start: coord[0]] + f'@{normalize_type(coord[1], is_ddi=is_ddi)}$'
        chunk_start = coord[0]
    output += raw_input[ordered_coordinates[-1][0]: len(raw_input)]

    return output


def normalize_type(x, is_ddi=False):
    if is_ddi:
        return 'DRUG'
    x = re.sub(r'[-_]\w', '', x)
    return x


def get_data_keys(dataset_name):
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
