from pathlib import Path

from tools.llm.deployments import *

HF_MODELS = [
    '01-ai/Yi-1.5-34B-Chat',
    'chaoyi-wu/MedLLaMA_13B',
    'epfl-llm/meditron-7b',
    'google/flan-t5-xxl',
    'HuggingFaceH4/zephyr-7b-beta',
    'instruction-pretrain/medicine-Llama3-8B',
    'meta-llama/Meta-Llama-3-8B-Instruct',
]
MODELS = HF_MODELS + AZURE_DEPLOYMENTS

BIOSSES, CHEMPROT, DDI, GAD, HOC, PUBMEDQA, BIOASQ = 'BIOSSES', 'ChemProt', 'DDI', 'GAD', 'HoC', 'PubmedQA', 'BioASQ'
DATASETS = [
    BIOSSES,
    CHEMPROT,
    DDI,
    GAD,
    HOC,
    PUBMEDQA,
    BIOASQ
]

SCORES_FILE_HEADERS = [
    'Model',
    'Dataset',
    'Prompt template',
    'Score',
    'Metric'
]
PREDICTIONS_FILE_HEADERS = [
    'Model',
    'Dataset',
    'Prompt template',
    'Example ID',
    'Prediction (raw)',
    'Valid?',
    'Prediction (resolved)',
    'Target'
]
PROMPTS_FILE_HEADERS = [
    'Model',
    'Dataset',
    'Prompt template',
    'Example ID',
    'Tokens',
    'Max tokens',
    'Valid?'
]
TEMP_DIR = Path(__file__).parent / 'tmp'
DEFAULT_OUTPUT_DIR = Path(__file__).parent / 'out'
SCORES_DIR = TEMP_DIR / 'scores'
PREDICTIONS_DIR = TEMP_DIR / 'predictions'
PROMPTS_DIR = TEMP_DIR / 'prompts'
SCORES_FILE = 'scores.csv'
PREDICTIONS_FILE = 'predictions.csv'
PROMPT_STATS_FILE = 'prompts.csv'
RESOURCES_DIR = Path(__file__).parent / 'resources'
EXAMPLE_MAP_PATH = RESOURCES_DIR / 'marked_selected_examples' / 'example_map.json'
RANDOM_EXAMPLE_MAP_PATH = RESOURCES_DIR / 'marked_selected_examples' / 'random_example_map.json'
HOC_LABEL_TO_ABBR_MAP_FILE = RESOURCES_DIR / 'hoc_label_to_abbr_map.json'
HOC_ABBR_TO_LABEL_MAP_FILE = RESOURCES_DIR / 'hoc_abbr_to_label_map.json'
MIN_VALID_PROMPTS = 0.98  # an experiment run must have 98% of valid prompts to be considered valid
INVALID_RUN = 'N/A'

EXAMPLE_OUTPUT_PLACEHOLDERS = {HOC: 'EMPTY_LIST'}
DEFAULT_PROMPT_CONFIG = Path(__file__).parent / 'configs' / 'config.yml'
CHAT_PROMPT_CONFIG = Path(__file__).parent / 'configs' / 'chat_config.yml'