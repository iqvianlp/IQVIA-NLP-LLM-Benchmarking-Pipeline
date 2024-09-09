import argparse
from pathlib import Path

from pipelines.benchmark_blurb_ner_and_pico import MARKED_SELECTED_EXAMPLES_DIR
from tools.bionlp_benchmarking.preprocess_blurb_raw_data import download_and_preprocess_blurb_data
from tools.example_mapping.fill_example_maps import (
    fill_examples_for_all_datasets, NER_AND_PICO_MAP_DIR, OTHER_TASK_MAP_DIR, OTHER_TASK_OUTPUT_DIR
)
from utils.logger import LOGGER


def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Performs all data preprocessing needed for the BLURB benchmarking'
                                                 ' scripts')
    return parser.parse_args()


def main():
    parse_args()
    LOGGER.info('--- Downloading and Preprocessing DDI and ChemProt Datasets ---')
    download_and_preprocess_blurb_data(['DDI', 'ChemProt'])
    LOGGER.info('--- Filling in Examples for all Datasets ---')
    fill_examples_for_all_datasets(
        Path(NER_AND_PICO_MAP_DIR),
        Path(MARKED_SELECTED_EXAMPLES_DIR),
        Path(OTHER_TASK_MAP_DIR),
        Path(OTHER_TASK_OUTPUT_DIR)
    )
    LOGGER.info('--- Done! ---')


if __name__ == '__main__':
    main()
