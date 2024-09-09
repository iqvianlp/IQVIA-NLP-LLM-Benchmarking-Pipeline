from pathlib import Path

from pipelines.benchmark_blurb_other_tasks.constants import PUBMEDQA
from tools.prompt_gen import get_prompt
from utils.io_utils import load_yaml

CONFIG = load_yaml(Path(__file__).parent / 'dummy_config.yml')


class TestGetPromptForPubmedQA:

    def test_with_short_zero_shot_prompt_check_all_good(self):
        self._assert_prompt(
            dataset=PUBMEDQA,
            template='short, zero-shot',
            expected=
            "Your task is to answer biomedical questions using the given abstract. Only output yes, no, or maybe as "
            "answer.\n"
            "Input: Question: input question Abstract: input abstract\n"
            "Output:"
        )

    def test_with_short_few_shot_prompt_check_all_good(self):
        self._assert_prompt(
            dataset=PUBMEDQA,
            template='short, few-shot (3)',
            expected=
            "Your task is to answer biomedical questions using the given abstract. Only output yes, no, or maybe as "
            "answer.\n"
            "Example:\n"
            "Input: Question: similar question 1 Abstract: similar abstract 1\n"
            "Output: answer 1\n"
            "Example:\n"
            "Input: Question: similar question 2 Abstract: similar abstract 2\n"
            "Output: answer 2\n"
            "Example:\n"
            "Input: Question: similar question 3 Abstract: similar abstract 3\n"
            "Output: answer 3\n"
            "Input: Question: input question Abstract: input abstract\n"
            "Output:"
        )

    def test_with_long_zero_shot_prompt_check_all_good(self):
        self._assert_prompt(
            dataset=PUBMEDQA,
            template='long, zero-shot',
            expected=
            "***INPUT***\n"
            "The input is a question followed by an abstract.\n"
            "***OUTPUT***\n"
            "Answer each question by providing one of the following options: yes, no, maybe.\n"
            "***YOUR TURN***\n"
            "Input: Question: input question Abstract: input abstract\n"
            "Output:"
        )

    def test_with_long_few_shot_prompt_check_all_good(self):
        self._assert_prompt(
            dataset=PUBMEDQA,
            template='long, few-shot (3)',
            expected=
            "***INPUT***\n"
            "The input is a question followed by an abstract.\n"
            "***OUTPUT***\n"
            "Answer each question by providing one of the following options: yes, no, maybe.\n"
            "***EXAMPLES***\n"
            "Example:\n"
            "Input: Question: similar question 1 Abstract: similar abstract 1\n"
            "Output: answer 1\n"
            "Example:\n"
            "Input: Question: similar question 2 Abstract: similar abstract 2\n"
            "Output: answer 2\n"
            "Example:\n"
            "Input: Question: similar question 3 Abstract: similar abstract 3\n"
            "Output: answer 3\n"
            "***YOUR TURN***\n"
            "Input: Question: input question Abstract: input abstract\n"
            "Output:"
        )

    def _assert_prompt(self, dataset, template, expected):
        example_id, example_map = 'dummy_id', self._dummy_example_map()
        actual = get_prompt(
            task=self._get_task(dataset),
            template=self._get_template(template),
            inputt=example_map[dataset][example_id]['input'],
            example_map=example_map,
            test_example_id=example_id
        )
        assert actual == expected

    @staticmethod
    def _dummy_example_map():
        return {
            PUBMEDQA: {
                'dummy_id': {
                    'input': [
                        'input question',
                        'input abstract'
                    ],
                    'sims': [
                        {
                            'input': [
                                'similar question 1',
                                'similar abstract 1'
                            ],
                            'output': 'answer 1'
                        },
                        {
                            'input': [
                                'similar question 2',
                                'similar abstract 2'
                            ],
                            'output': 'answer 2'
                        },
                        {
                            'input': [
                                'similar question 3',
                                'similar abstract 3'
                            ],
                            'output': 'answer 3'
                        }
                    ]
                }
            }
        }

    @staticmethod
    def _get_task(dataset):
        for t in CONFIG['tasks']:
            if t['dataset'] == dataset:
                return t
        raise ValueError(f'Task for dataset "{dataset}" does not exist in config.')

    @staticmethod
    def _get_template(name):
        for t in CONFIG['templates']:
            if t['name'] == name:
                return t
        raise ValueError(f'Prompt template with name "{name}" does not exist in config.')
