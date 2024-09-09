"""
This script is derived from the official BLURB preprocessing script
(https://microsoft.github.io/BLURB/sample_code/data_generation.tar.gz).

MIT License

Copyright (c) 2020 Microsoft

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import argparse
import re
import os
import shutil
import tarfile
import urllib.request
import zipfile
from typing import List
from xml.etree import cElementTree as ET

import nltk
from nltk.tokenize import sent_tokenize

from utils.io_utils import save_json
from utils.logger import LOGGER

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "raw")
PROCESSED_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "blurb", "evaluators", "resources", "processed"
)

RAW_DATA_URLS = {
    'DDI': r'https://github.com/zhangyijia1979/hierarchical-RNNs-model-for-DDI-extraction/raw/master/DDIextraction2013/DDIextraction_2013.tar.gz',
    'ChemProt': r'https://biocreative.bioinformatics.udel.edu/media/store/files/2017/ChemProt_Corpus.zip'
}

VALID_DATASETS = ['DDI', 'ChemProt']

nltk.download('punkt')


def parse_args():
    parser = argparse.ArgumentParser(description='To pre-process DDI and ChemProt raw data for BLURB.')
    parser.add_argument(
        '--datasets',
        '-d',
        nargs="+",
        default=VALID_DATASETS,
        type=str,
        choices=VALID_DATASETS,
        help='The dataset to pre-process.'
    )
    return parser.parse_args()


def customized_sent_tokenize(text):
    sents_raw = sent_tokenize(text)
    output_sents = []
    for sent in sents_raw:
        if len(sent.split('\t')) > 1:
            output_sents.extend(sent.split('\t'))
        else:
            output_sents.append(sent)
    return output_sents


def split_sent(e1_span_s, e1_span_e, e2_span_s, e2_span_e, sent):
    # if e1 e2 not overlaping, output 5 chunks; else output 3
    pos_list = [e1_span_s, e1_span_e, e2_span_s, e2_span_e]
    if e1_span_e > e2_span_s:
        entity_s = min(e1_span_s, e2_span_s)
        entity_e = max(e1_span_e, e2_span_e)
        pos_list = [entity_s, entity_e]
    # if pos_list != sorted(pos_list):
    #     raise ValueError("Positions not in order!")
    spans = zip([0] + pos_list, pos_list + [len(sent)])
    output_chunks = []
    for (s, e) in spans:
        output_chunks.append(sent[s:e])
    return output_chunks


def extract_relation_dict(relations_file, target_labels):
    # Forming relation reference dictionary
    # {doc_id:{(e1, e2): label}}
    with open(relations_file, 'r', encoding='utf-8', errors='replace') as f:
        relations = f.readlines()

    relation_ref_dict = {}
    for line in relations:
        doc_id, label, _, _, e1, e2 = line.rstrip().split('\t')
        e1_id = e1.split(':')[1]
        e2_id = e2.split(':')[1]
        if doc_id not in relation_ref_dict:
            relation_ref_dict[doc_id] = {}
        label = label if label in target_labels else "false"
        relation_ref_dict[doc_id][(e1_id, e2_id)] = label
    return relation_ref_dict


def extract_entity_dict(entities_file):
    # entity span refer
    # {doc_id:[[e_id, type, span_s, span_e, content]]}
    with open(entities_file, 'r', encoding='utf-8', errors='replace') as f:
        entities = f.readlines()
    entity_span_dict = {}
    for line in entities:
        doc_id, e_id, type_, span_s, span_e, content = line.rstrip().split('\t')
        if doc_id not in entity_span_dict:
            entity_span_dict[doc_id] = []
        # Ignoring the suffixes
        type_ = type_.split('-')[0]
        entity_span_dict[doc_id].append(
            [e_id, type_, int(span_s), int(span_e), content])
    return entity_span_dict


def reformat_data(abstract_file, relation_ref_dict, entity_span_dict, next_document_id):
    details = list()
    # Traversing abstract, and finding candidates with exact one chem
    # and one gene
    with open(abstract_file, 'r', encoding='utf-8', errors='replace') as f:
        abstract_data = f.readlines()

    for line in abstract_data:
        doc_id, text = line.split('\t', 1)
        sents = customized_sent_tokenize(text)
        entity_candidates = entity_span_dict[doc_id]
        prev_span_end = 0
        for sent in sents:
            # Extracting span of cur sent.
            sent_span_s = text.find(sent, prev_span_end)
            sent_span_e = sent_span_s + len(sent)
            prev_span_end = sent_span_e
            chem_list = []
            gene_list = []
            for entity_candidate in entity_candidates:
                e_id, type_, entity_span_s, entity_span_e, content = \
                    entity_candidate
                if entity_span_s >= sent_span_s and entity_span_e \
                        <= sent_span_e:
                    if "CHEMICAL" in type_:
                        chem_list.append(entity_candidate)
                    else:
                        gene_list.append(entity_candidate)
            if len(chem_list) == 0 or len(gene_list) == 0:
                continue

            details.append(
                {
                    'document_id': str(next_document_id),
                    'text': sent,
                    'entities': {
                        'id': [item[0] for item in chem_list] + [item[0] for item in gene_list],
                        'type': [item[1] for item in chem_list] + [item[1] for item in gene_list],
                        'text': [item[4] for item in chem_list] + [item[4] for item in gene_list],
                        'offsets':
                            [[item[2] - sent_span_s, item[3] - sent_span_s] for item in chem_list] +
                            [[item[2] - sent_span_s, item[3] - sent_span_s] for item in gene_list]
                    },
                    'relations': {
                        'type': list(),
                        'arg1': list(),
                        'arg2': list()
                    }
                }
            )
            next_document_id += 1
            # Preparing data with appending method
            for chem_candidate in chem_list:
                for gene_candidate in gene_list:

                    # Denoting the first entity entity 1.
                    if chem_candidate[2] < gene_candidate[2]:
                        e1_candidate, e2_candidate = \
                            chem_candidate, gene_candidate
                    else:
                        e2_candidate, e1_candidate = \
                            chem_candidate, gene_candidate
                    e1_id, e1_type, e1_span_s, e1_span_e, e1_content = \
                        e1_candidate
                    e2_id, e2_type, e2_span_s, e2_span_e, e2_content = \
                        e2_candidate
                    label = "false"

                    if doc_id in relation_ref_dict:
                        if (e1_id, e2_id) in relation_ref_dict[doc_id]:
                            label = relation_ref_dict[doc_id][(e1_id, e2_id)]
                        elif (e2_id, e1_id) in relation_ref_dict[doc_id]:
                            label = relation_ref_dict[doc_id][(e2_id, e1_id)]

                    details[-1]['relations']['type'].append(label)
                    details[-1]['relations']['arg1'].append(e1_id)
                    details[-1]['relations']['arg2'].append(e2_id)

    return details, next_document_id


def clean_sent(sent):
    special_chars = ['\n', '\t', '\r']
    for special_char in special_chars:
        sent = sent.replace(special_char, ' ')
    return sent


def extract_entity_dict_ddi(entity_candidates):
    candidate_ref_dict = {}
    for entity_candidate in entity_candidates:
        candidate_ref_dict[entity_candidate.get(
            'id')] = entity_candidate.attrib
    return candidate_ref_dict


def extract_span_ddi(span_str):
    candidates = re.findall(r"\d+", span_str)
    # When multiple spans occurs, only taking the very first and last positions
    # Ending position offsets by 1
    span_s, span_e = int(candidates[0]), int(candidates[-1]) + 1

    return [span_s, span_e]


def dump_processed_data(output_dir: str, data_type: str, details: list):
    filepath = os.path.join(output_dir, f"{data_type}.json")
    os.makedirs(output_dir, exist_ok=True)
    save_json(details, filepath, indent=4)


def prepare_chemprot_data(root_dir, output_dir):
    data_types = ['train', 'dev', 'test']
    target_labels = ["CPR:3", "CPR:4", "CPR:5", "CPR:6", "CPR:9"]

    next_document_id = 1
    for data_type in data_types:
        if data_type == 'train':
            data_path = os.path.join(
                root_dir, 'chemprot_training/')
            file_name_prefix = "chemprot_training_"
            file_name_affix = ""
        elif data_type == "dev":
            data_path = os.path.join(
                root_dir, 'chemprot_development/')
            file_name_prefix = "chemprot_development_"
            file_name_affix = ""
        else:
            data_path = os.path.join(
                root_dir, 'chemprot_test_gs/')
            file_name_prefix = "chemprot_test_"
            file_name_affix = "_gs"

        relations_file = os.path.join(
            data_path, f"{file_name_prefix}relations{file_name_affix}.tsv")
        entities_file = os.path.join(
            data_path, f"{file_name_prefix}entities{file_name_affix}.tsv")
        abstract_file = os.path.join(
            data_path, f"{file_name_prefix}abstracts{file_name_affix}.tsv")

        relation_ref_dict = extract_relation_dict(
            relations_file, target_labels)
        entity_span_dict = extract_entity_dict(entities_file)
        (details, next_document_id) = reformat_data(
            abstract_file, relation_ref_dict, entity_span_dict, next_document_id
        )
        # Dumping data.
        dump_processed_data(output_dir=output_dir, data_type=data_type, details=details)


def prepare_ddi_data(root_dir, output_dir):
    data_types = ["train", "dev", "test"]
    # prepare train/dev/test file names

    next_document_id = 1
    for data_type in data_types:
        # load file names
        with open(
                os.path.join(RAW_DATA_DIR, 'indexing', 'ddi', f'{data_type}_files.tsv'),
                'r', encoding='utf-8', errors='replace'
        ) as f:
            data_file_names = f.readlines()
        data_file_paths = [os.path.join(root_dir, x.strip()) for x in data_file_names]
        details = list()
        for data_file_path in data_file_paths:
            tree = ET.parse(data_file_path)
            root = tree.getroot()
            for sent in list(root):
                ids_in_details = list()
                if sent.find('pair') is None:
                    continue
                text = sent.attrib.get('text')
                details.append(
                    {
                        'document_id': str(next_document_id),
                        'text': text,
                        'entities': list(),
                        'relations': list()
                    }
                )
                next_document_id += 1
                entity_candidates = sent.findall('entity')
                candidate_ref_dict = {}
                for entity_candidate in entity_candidates:
                    candidate_ref_dict[entity_candidate.get(
                        'id')] = entity_candidate.attrib
                pairs = sent.findall('pair')
                for pair in pairs:
                    pair_id, e1_id, e2_id, label = pair.attrib['id'], pair.attrib[
                        'e1'], pair.attrib['e2'], pair.attrib['ddi']
                    if label == 'true':
                        label = pair.attrib['type']
                    e1 = candidate_ref_dict[e1_id]
                    e2 = candidate_ref_dict[e2_id]

                    # Ensuring e1 is the first entity.
                    if extract_span_ddi(e1['charOffset'])[0] > extract_span_ddi(e2['charOffset'])[0]:
                        e1, e2 = e2, e1
                    e1_span_s, e1_span_e = extract_span_ddi(e1['charOffset'])
                    e2_span_s, e2_span_e = extract_span_ddi(e2['charOffset'])
                    if e1_id not in ids_in_details:
                        ids_in_details.append(e1_id)
                        details[-1]['entities'].append(
                            {
                                'offsets': [e1_span_s, e1_span_e],
                                'text': text[e1_span_s: e1_span_e],
                                'type': 'DRUG',
                                'id': f'T{e1_id.split(".")[-1][1:]}'
                            }
                        )
                    if e2_id not in ids_in_details:
                        ids_in_details.append(e2_id)
                        details[-1]['entities'].append(
                            {
                                'offsets': [e2_span_s, e2_span_e],
                                'text': text[e2_span_s: e2_span_e],
                                'type': 'DRUG',
                                'id': f'T{e2_id.split(".")[-1][1:]}'
                            }
                        )
                    details[-1]['relations'].append(
                        {
                            'id': f'R{pair_id.split(".")[-1][1:]}',
                            'head': {
                                'ref_id': f'T{e1_id.split(".")[-1][1:]}',
                                'role': 'Arg1'
                            },
                            'tail': {
                                'ref_id': f'T{e2_id.split(".")[-1][1:]}',
                                'role': 'Arg2'
                            },
                            'type': label.upper()
                        }
                    )

        dump_processed_data(output_dir=output_dir, data_type=data_type, details=details)


def download_ddi_data(dataset_name: str):
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    url = RAW_DATA_URLS[dataset_name]
    local_zip_path = os.path.join(RAW_DATA_DIR, f'{dataset_name}.tar.gz')
    urllib.request.urlretrieve(url, local_zip_path)
    with tarfile.open(local_zip_path, "r") as z:
        z.extractall(RAW_DATA_DIR)


def download_chemprot_data(dataset_name: str):
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    url = RAW_DATA_URLS[dataset_name]
    local_zip_path = os.path.join(RAW_DATA_DIR, f'{dataset_name}.zip')
    urllib.request.urlretrieve(url, local_zip_path)
    with zipfile.ZipFile(local_zip_path, "r") as z:
        z.extractall(RAW_DATA_DIR)
    extracted_dir = os.path.join(RAW_DATA_DIR, 'ChemProt_Corpus')
    train_zip_path = os.path.join(extracted_dir, 'chemprot_training.zip')
    dev_zip_path = os.path.join(extracted_dir, 'chemprot_development.zip')
    test_zip_path = os.path.join(extracted_dir, 'chemprot_test_gs.zip')
    for path in (train_zip_path, dev_zip_path, test_zip_path):
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(extracted_dir)


def download_and_preprocess_blurb_data(datasets: List[str]):
    for dataset in datasets:

        LOGGER.info(f'Downloading {dataset} raw data...')
        if dataset == 'DDI':
            download_ddi_data(dataset)
        else:
            download_chemprot_data(dataset)

        LOGGER.info(f'Preprocessing {dataset} data...')
        ddi_unzipped_dir = os.path.join(RAW_DATA_DIR, 'DDIextraction_2013')
        chemprot_unzipped_dir = os.path.join(RAW_DATA_DIR, 'ChemProt_Corpus')
        prepare_ddi_data(
            root_dir=ddi_unzipped_dir,
            output_dir=os.path.join(PROCESSED_DATA_DIR, f'{dataset}_preprocessed')
        ) if dataset == 'DDI' else prepare_chemprot_data(
            root_dir=chemprot_unzipped_dir,
            output_dir=os.path.join(PROCESSED_DATA_DIR, f'{dataset}_preprocessed')
        )

        LOGGER.info(f"Clear local unzipped {dataset} raw data...")
        if os.path.exists(ddi_unzipped_dir):
            shutil.rmtree(ddi_unzipped_dir)
        if os.path.exists(chemprot_unzipped_dir):
            shutil.rmtree(chemprot_unzipped_dir)


def main():
    args = parse_args()
    download_and_preprocess_blurb_data(args.datasets)


if __name__ == "__main__":
    main()
