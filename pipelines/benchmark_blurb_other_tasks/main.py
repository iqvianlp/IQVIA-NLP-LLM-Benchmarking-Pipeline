import argparse
import ast
import csv
from io import StringIO
import statistics

from pandas import DataFrame, read_csv, concat
from tqdm import tqdm
from nltk.tokenize import wordpunct_tokenize

from pipelines.benchmark_blurb_other_tasks.common import (
    load_hoc_maps,
    load_evaluator,
    filter_data,
    augment_entry,
    hoc_ints_to_str
)
from pipelines.benchmark_blurb_other_tasks.constants import *
from tools.prompt_gen import get_prompt, get_chat_prompt
from utils.io_utils import load_yaml, load_json, save_txt, load_txt, save_json
from utils.logger import base_logger
from utils.text_utils import clean_str
from tools.llm.clients import AzureClient, TGIClient, VLLMClient


def _parse_args():
    parser = argparse.ArgumentParser(description='BLURB evaluator main script.')
    parser.add_argument(
        '--ip',
        '-i',
        required=False,
        help=f'IP address or URL of LLM server.')
    parser.add_argument(
        '--output_dir',
        '-o',
        required=False,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory.')
    parser.add_argument(
        '--example_map_file',
        '-e',
        required=False,
        default=EXAMPLE_MAP_PATH,
        help='Path of the example map file.')
    parser.add_argument(
        '--models',
        '-m',
        nargs='+',
        choices=MODELS,
        required=True,
        help='The names of models to use. Must be one or more of: ' + ', '.join(MODELS))
    parser.add_argument(
        '--datasets',
        '-d',
        nargs='+',
        choices=DATASETS,
        required=False,
        help='The names of datasets to use. Must be one or more of: ' + ', '.join(DATASETS))
    parser.add_argument(
        '--limit',
        '-l',
        type=int,
        required=False,
        default=None,
        help='The max number of test examples for each run. Note that this is ignored for some datasets, as all '
             'examples are required for them.')
    parser.add_argument(
        '--overwrite',
        '-ow',
        action='store_true',
        required=False,
        help='Whether to re-run scenarios with existing output files and overwrite those files.'
    )
    parser.add_argument(
        '--client',
        '-cl',
        default='tgi',
        choices=['tgi', 'vllm'],
        help='Type of client to use. Should match the type of server used for the `--ip` parameter. Does not affect'
             ' OpenAI models.'
    )
    parser.add_argument(
        '--chat-completion',
        '-cc',
        action='store_true',
        required=False,
        help='Whether to use chat completion prompts when prompting models. '
    )
    parser.add_argument(
        '--use-system-role',
        '-usr',
        action='store_true',
        required=False,
        help='Whether to provide initial instructions to the LLM using the "system" role when chat completion is '
             'enabled.'
    )
    return parser.parse_args()


class Runner:
    """Experiment runner.

    Loads 3 maps:
        1. HOC labels to abbreviations
        2. HOC abbreviations to labels
        3. Example maps for all datasets
    """

    def __init__(self, config, models, output_dir, example_map_file, datasets, limit, ip,
                 overwrite: bool = False, client: str = 'tgi'):
        self.config = config
        self.models = models
        self.temp_scores_dir = SCORES_DIR
        self.temp_predictions_dir = PREDICTIONS_DIR
        self.temp_prompt_stats_dir = PROMPTS_DIR
        self.output_dir = Path(output_dir)
        self.scores_file = self.output_dir / SCORES_FILE
        self.predictions_file = self.output_dir / PREDICTIONS_FILE
        self.prompt_stats_file = self.output_dir / PROMPT_STATS_FILE
        self.prompts_dir = self.output_dir / 'prompts'
        self.datasets = datasets
        self.limit = limit
        self.ip = ip
        self.logger = base_logger()
        if example_map_file is not None:
            self.example_map = self._load_example_map(example_map_file)
        self.hoc_label_to_abbr_map, _ = load_hoc_maps()
        self._overwrite = overwrite
        self._client = client

    def run(self, chat_completion_enabled: bool = False, use_system_role: bool = False):
        """Runs the whole experiment with the supplied configuration.

        Experiment steps are:
            - For each model-dataset-prompt-template combo...
            - For each entry in the test split of the dataset...
            - If ChemProd or DDI, repeat below steps for each relation
            - Populate prompt template with entry (annotated if ChemProd or DDI)
            - If not zero shot, populate template also with best examples for entry
            - Get prediction for the populated prompt from the model
            - Store score in a predictions dictionary with the right key (as defined by the task)
            - Compute overall score for the combo and the predictions dictionary
            - Combine all scores in a data frame
            - Save scores and predictions to disk (and print scores to console)
        """
        self.logger.info(f'Experiment started with:\n'
                         f'  - models: {", ".join(self.models)}\n'
                         f'  - datasets: {", ".join(self.datasets)}\n'
                         f'  - limit: {self.limit}\n'
                         f'  - output dir: {self.output_dir.resolve()}'
                         )

        scenario = 1
        for model in self.models:
            client = self._load_llm_client(model, chat_completion_enabled)
            if not self._can_run_model(client):
                continue
            else:
                for task in self.config['tasks']:
                    if task['dataset'] in self.datasets:
                        evaluator = load_evaluator(task['dataset'])
                        test_set = self._get_test_set(task['dataset'], evaluator)
                        for template in self.config['templates']:
                            self.logger.info(f'Scenario {scenario}: {task["dataset"]}; {template["name"]}; {model}')
                            scenario += 1
                            if not self._overwrite and self._scenario_done(model, task['dataset'], template['name']):
                                self.logger.info('Scenario already done; skipping it')
                                continue
                            else:
                                stats = self._get_prompt_stats(
                                    client, test_set, task['dataset'], task, template, use_system_role
                                )
                                if stats['skip']:
                                    self._skip_scenario(task['dataset'], template['name'], stats, client.model_name)
                                else:
                                    self._run_scenario(task['dataset'], template['name'], stats, client, evaluator)
        self._aggregate_results()
        self.logger.info('Done!')

    def _load_example_map(self, example_map_file):
        example_map_path = Path(example_map_file)
        if example_map_path.exists():
            self.logger.info(f'Loading example map at {example_map_path.resolve()}')
            example_map = load_json(example_map_path)
            diff = set(self.datasets) - set(example_map.keys())
            if len(diff) > 0:
                self.logger.error(f'Stopping experiment as example map is missing some datasets: {", ".join(diff)}. '
                                  f'Build example map with run_data_prep.py before running the experiment')
                exit(1)
            return example_map
        else:
            raise FileNotFoundError(f'{example_map_path.resolve()} not found; build example map with run_data_prep.py'
                                    f' before running the experiment')

    def _load_llm_client(self, model_name: str, chat_completion_enabled: bool):
        self.logger.info(f'Loading LLM client for "{model_name}"')
        if model_name in AZURE_DEPLOYMENTS:
            llm_client = AzureClient(model_name, chat_completion_enabled=chat_completion_enabled)
        else:
            if self._client == 'tgi':
                llm_client = TGIClient(self.ip, model_name, chat_completion_enabled=chat_completion_enabled)
            else:
                llm_client = VLLMClient(self.ip, model_name, chat_completion_enabled=chat_completion_enabled)
        self.logger.info(f'LLM client loaded: {llm_client}')
        return llm_client

    def _can_run_model(self, llm_client):
        if llm_client.model_name not in AZURE_DEPLOYMENTS and self.ip is None:
            self.logger.warning(f'Skipping non-Azure model "{llm_client.model_name}" as no IP address was provided; '
                                f'run script with -i [URL/IP address of endpoint serving this model]')
            return False
        if not llm_client.is_alive():
            self.logger.warning(f'LLM server can\'t be reached! Skipping "{llm_client.model_name}"')
            return False
        self.logger.info(f'LLM server is alive. Proceed with experiment with model {llm_client.model_name}')
        return True

    def _get_test_set(self, dataset_name, evaluator):
        test_set = evaluator.load_split(split='test')
        test_set = filter_data(self.logger, dataset_name, test_set)
        # Convert all example IDs to str as this is how the example map saves them:
        test_set = {str(k): v for k, v in test_set.items()}
        if self.limit and dataset_name not in [BIOSSES, GAD, PUBMEDQA, BIOASQ]:  # some evaluators need all examples
            return dict(list(test_set.items())[:self.limit])
        return test_set

    def _scenario_done(self, model_name, dataset_name, template_name):
        csv_name = self._get_scenario_hash(model_name, dataset_name, template_name)
        return Path(self.temp_scores_dir, csv_name).with_suffix('.csv').exists()

    def _get_prompt_stats(self, llm_client, test_set, dataset_name, task, template, use_system_role):
        self.logger.info(f'Computing prompt stats')
        total_prompts, valid_prompts, paths, example_ids, targets, prompt_stats = 0, 0, [], [], [], []
        for key in tqdm(test_set):
            for target, entry in self._get_target_and_entry(dataset_name, test_set[key]):
                total_prompts += 1
                example_id = self._get_key(dataset_name, entry)
                prompt_path = Path(
                    self.prompts_dir,
                    clean_str(dataset_name),
                    clean_str(template['name']),
                    clean_str(example_id)
                ).with_suffix('.txt' if not llm_client.chat_completion_enabled else '.json')
                if not self._overwrite and prompt_path.exists():
                    prompt = load_txt(prompt_path) if not llm_client.chat_completion_enabled else load_json(prompt_path)
                else:
                    inputt = self._get_input(dataset_name, entry)
                    if not llm_client.chat_completion_enabled:
                        prompt = get_prompt(task, template, inputt, self.example_map, key)
                        save_txt(prompt, prompt_path, parents=True)
                    else:
                        prompt = get_chat_prompt(
                            task,
                            template,
                            inputt,
                            self.example_map,
                            key,
                            '' if dataset_name not in EXAMPLE_OUTPUT_PLACEHOLDERS else
                            EXAMPLE_OUTPUT_PLACEHOLDERS[dataset_name],
                            use_system_role
                        )
                        save_json(prompt, prompt_path, parents=True)
                example_ids.append(example_id)
                targets.append(target)
                tokens, is_valid = llm_client.is_valid_prompt(prompt)
                if is_valid:
                    paths.append(prompt_path)
                    valid_prompts += 1
                prompt_stats.append([
                    llm_client.model_name,
                    dataset_name,
                    template['name'],
                    example_id,
                    tokens,
                    llm_client.model_max_length,
                    is_valid
                ])
        self._save_prompt_stats(
            llm_client.model_name,
            dataset_name,
            template['name'],
            prompt_stats
        )
        min_valid_prompts = 1 if dataset_name in [BIOSSES, GAD, PUBMEDQA, BIOASQ] else MIN_VALID_PROMPTS
        return {
            'total': total_prompts,
            'valid': valid_prompts,
            'invalid': total_prompts - valid_prompts,
            'ratio': valid_prompts / total_prompts,
            'skip': (valid_prompts / total_prompts) < min_valid_prompts,
            'paths': paths,
            'example_ids': example_ids,
            'targets': targets
        }

    def _load_saved_prompt_stats(self, dataset_name: str, template_name: str, model_name: str) -> dict:
        """
        Loads prompt stats for the given dataset/prompt template combination from the saved aggregated table of stats.

        Args:
            dataset_name: Dataset name.
            template_name: Prompt template name.
            model_name: Model name.

        Returns:
            dict[str, Any]: A dict of prompt stats.
        """
        ps_df = read_csv(self.prompt_stats_file, na_filter=False, dtype=str)
        stats = ps_df[
            (ps_df['Model'] == model_name) &
            (ps_df['Dataset'] == dataset_name) &
            (ps_df['Prompt template'] == template_name)
        ][['Example ID', 'Valid?']]
        total_prompts = len(stats)
        valid_prompts = len(stats[stats['Valid?'] == 'True'])
        min_valid_prompts = 1 if dataset_name in [BIOSSES, GAD, PUBMEDQA, BIOASQ] else MIN_VALID_PROMPTS
        return {
            'total': total_prompts,
            'valid': valid_prompts,
            'invalid': total_prompts - valid_prompts,
            'ratio': valid_prompts / total_prompts,
            'skip': (valid_prompts / total_prompts) < min_valid_prompts,
        }

    @staticmethod
    def _get_target_and_entry(dataset_name, entry):
        if dataset_name == BIOSSES:
            yield statistics.fmean([entry[f'annotator_{x}'] for x in ['a', 'b', 'c', 'd', 'e']]), entry
        elif dataset_name == CHEMPROT:
            for i, pair in enumerate(zip(entry['relations']['arg1'], entry['relations']['arg2'])):
                yield entry['relations']['type'][i], augment_entry(entry, pair)
        elif dataset_name == DDI:
            for relation in entry['relations']:
                pair = (relation['head']['ref_id'], relation['tail']['ref_id'])
                yield relation['type'], augment_entry(entry, pair, is_ddi=True)
        elif dataset_name == GAD:
            yield entry['label'], entry
        elif dataset_name == HOC:
            target = entry['label']
            if 7 in target:
                target.remove(7)  # 7 = no hallmarks of cancer and is not considered when evaluating
            yield hoc_ints_to_str(target), entry
        elif dataset_name == PUBMEDQA:
            yield entry['answer'], entry
        else:  # BioASQ
            yield entry['exact_answer'][0], entry

    @staticmethod
    def _get_input(dataset_name, entry):
        if dataset_name == BIOSSES:
            return [entry['text_1'], entry['text_2']]
        elif dataset_name == CHEMPROT:
            return [entry['text_annotated']]  # dynamically generated for each sub-example of an example (row)
        elif dataset_name == DDI:
            return [entry['text_annotated']]  # dynamically generated for each sub-example of an example (row)
        elif dataset_name == GAD:
            return [entry['sentence']]
        elif dataset_name == HOC:
            return [entry['text']]
        elif dataset_name == PUBMEDQA:
            return [entry['question'], entry['context']]
        else:  # BioASQ
            return [entry['body']]

    @staticmethod
    def _get_key(dataset_name, entry):
        if dataset_name == BIOSSES:
            key = entry['document_id']
        elif dataset_name == CHEMPROT:
            key = entry['example_id']  # dynamically generated for each sub-example of an example (row)
        elif dataset_name == DDI:
            key = entry['example_id']  # dynamically generated for each sub-example of an example (row)
        elif dataset_name == GAD:
            key = entry['index']
        elif dataset_name == HOC:
            key = entry['document_id']
        elif dataset_name == PUBMEDQA:
            key = entry['question_id']
        else:  # BioASQ
            key = entry['id']
        return str(key)

    def _save_prompt_stats(self, model_name, dataset_name, template_name, prompt_stats):
        csv_name = self._get_scenario_hash(model_name, dataset_name, template_name)
        prompt_path = Path(self.temp_prompt_stats_dir, csv_name).with_suffix('.csv')
        prompt_df = DataFrame(prompt_stats, columns=PROMPTS_FILE_HEADERS)
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_df.to_csv(prompt_path, index=False)

    @staticmethod
    def _get_scenario_hash(model_name, dataset_name, template_name, extras=None):
        parts = [model_name, dataset_name, template_name]
        if extras is not None and isinstance(extras, list):
            parts += extras
        return '_'.join([clean_str(x) for x in parts])

    def _skip_scenario(self, dataset_name, template_name, prompt_stats, model_name):
        self.logger.warning(
            f'Excluding scenario with only '
            f'{(prompt_stats["ratio"]) * 100:.1f}% '
            f'({prompt_stats["valid"]}/{prompt_stats["total"]}) '
            f'valid prompts'
        )
        self._save_score_row(model_name, dataset_name, template_name, INVALID_RUN)

    def _save_score_row(self, model_name, dataset_name, template_name, score):
        csv_name = self._get_scenario_hash(model_name, dataset_name, template_name)
        scores_path = Path(self.temp_scores_dir, csv_name).with_suffix('.csv')
        self.logger.info(f'Scenario done; saving scores')
        score_row = [
            [
                model_name,
                dataset_name,
                template_name,
                score,
                self._get_metric(dataset_name).replace('_', ' ').title()
            ]
        ]
        scores_df = DataFrame(score_row, columns=SCORES_FILE_HEADERS)
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        scores_df.to_csv(scores_path, index=False)

    @staticmethod
    def _get_metric(dataset_name):
        if dataset_name == BIOSSES:
            metric = 'pearson'
        elif dataset_name == CHEMPROT:
            metric = 'micro_f1'
        elif dataset_name == DDI:
            metric = 'micro_f1'
        elif dataset_name == GAD:
            metric = 'micro_f1'
        elif dataset_name == HOC:
            metric = 'average_micro_f1'
        elif dataset_name == PUBMEDQA:
            metric = 'accuracy'
        else:  # BioASQ
            metric = 'accuracy'

        return metric

    def _run_scenario(self, dataset_name, template_name, prompt_stats, llm_client, evaluator):
        self.logger.info(
            f'Scenario is valid with '
            f'{(prompt_stats["ratio"]) * 100:.1f}% '
            f'({prompt_stats["valid"]}/{prompt_stats["total"]}) '
            f'valid prompts. Collecting predictions...')

        valid_responses = self._get_valid_responses(dataset_name, evaluator)
        predictions = {
            'dataset_name': dataset_name,
            'model_name': llm_client.model_name,
            'predictions': {}
        }
        prediction_rows = []
        for i, prompt_path in enumerate(tqdm(prompt_stats['paths'])):
            prompt = load_txt(prompt_path) if not llm_client.chat_completion_enabled else \
                load_json(prompt_path)
            example_id = prompt_stats['example_ids'][i]
            target = prompt_stats['targets'][i]
            raw_prediction, is_valid, prediction = (
                self._get_prediction(dataset_name, example_id, llm_client, predictions, valid_responses, prompt)
            )
            prediction_rows.append([
                llm_client.model_name,
                dataset_name,
                template_name,
                example_id,
                raw_prediction,
                is_valid,
                prediction,
                target
            ])
        self._save_predictions(llm_client.model_name, dataset_name, template_name, prediction_rows)
        score = self._get_score(evaluator, predictions, dataset_name)
        self._save_score_row(llm_client.model_name, dataset_name, template_name, score)

    def _eval_scenario_from_saved_preds(
            self,
            dataset_name: str,
            template_name: str,
            model_name: str,
            evaluator
    ) -> None:
        """
        Evaluates a scenario using saved predictions.

        Args:
            dataset_name: Dataset name.
            template_name: Template name.
            model_name: Model name.
            evaluator: BLURB task evaluator.

        Returns:
            None
        """
        predictions = self._load_saved_predictions(dataset_name, template_name, model_name)
        score = self._get_score(evaluator, predictions, dataset_name)
        self._save_score_row(model_name, dataset_name, template_name, score)

    @staticmethod
    def _get_valid_responses(dataset_name, evaluator):
        pp = evaluator._schema['properties']['predictions']['patternProperties']
        if dataset_name == BIOSSES:
            return {
                'min': pp['^[0-9]+$']['minimum'],
                'max': pp['^[0-9]+$']['maximum']
            }
        elif dataset_name == CHEMPROT:
            return evaluator._valid_relations
        elif dataset_name == DDI:
            return pp['^.+_T[0-9]+_T[0-9]+$']['enum']
        elif dataset_name == GAD:
            return pp['^[0-9]+$']['enum']
        elif dataset_name == HOC:
            return pp['^[0-9]+_[0-9]+$']['items']['enum']
        elif dataset_name == PUBMEDQA:
            return pp['^[0-9]{7,8}$']['enum']
        else:  # BioASQ
            return pp['^[a-f0-9]{24}$']['enum']

    def _get_prediction(self, dataset_name, example_id, llm_client, predictions_dict, valid_responses, prompt):
        try:
            raw_prediction = llm_client(prompt)
        except Exception as e:
            print(prompt)
            print(e)
            raw_prediction = '__ERROR__'
        is_valid, prediction = self._parse_prediction(raw_prediction, dataset_name, valid_responses)
        predictions_dict['predictions'][example_id] = prediction

        return raw_prediction, is_valid, prediction

    def _parse_prediction(self, prediction, dataset_name, valid_responses):
        """
        Attempts to parse the raw prediction into a valid format, returning the default prediction for that dataset if
        it fails.

        Behavior is dependent on the dataset:
            - BIOSSES: Takes the first word, stripping punctuation.
            - ChemProt: Takes the first word, stripping punctuation.
            - DDI: Takes the first word, stripping punctuation.
            - GAD: Takes the first word, stripping punctuation.
            - HoC: Parses the prediction as a CSV, searching for all HoC classes in the resulting list.
            - PubMedQA: Takes the first word (using wordpunct tokenization), and lowercases it.
            - BioASQ: Takes the first word (using wordpunct tokenization), and lowercases it.
        """
        prediction = prediction.strip()
        if dataset_name == BIOSSES:
            try:
                prediction = self._get_first_word(prediction)
                resolved = float(prediction)
                is_valid = True
                if resolved < valid_responses['min'] or resolved > valid_responses['max']:
                    resolved = 0.0
                    is_valid = False
            except ValueError:
                resolved = 0.0
                is_valid = False
        elif dataset_name == CHEMPROT:
            prediction = self._get_first_word(prediction)
            is_valid = prediction in valid_responses
            resolved = prediction if is_valid else 'false'
        elif dataset_name == DDI:
            prediction = self._get_first_word(prediction)
            is_valid = prediction in valid_responses
            resolved = prediction if is_valid else 'DDI-false'
        elif dataset_name == GAD:
            prediction = self._get_first_word(prediction)
            is_valid = prediction.isnumeric() and prediction in valid_responses
            resolved = prediction if is_valid else '0'
        elif dataset_name == HOC:
            resolved = []
            try:
                categories = next(csv.reader(StringIO(prediction), delimiter=','))
                categories = [x.strip().lower() for x in categories]
                is_valid = True
                for c in categories:
                    if c in self.hoc_label_to_abbr_map:
                        resolved.append(self.hoc_label_to_abbr_map[c])
                resolved = list(set(resolved))
            except csv.Error:
                is_valid = False
            except StopIteration:
                is_valid = False
        elif dataset_name == PUBMEDQA:
            try:
                prediction = wordpunct_tokenize(prediction)[0].lower()
            except IndexError:
                prediction = ''
            is_valid = prediction in valid_responses
            resolved = prediction if is_valid else 'maybe'
        else:  # BioASQ
            try:
                prediction = wordpunct_tokenize(prediction)[0].lower()
            except IndexError:
                prediction = ''
            is_valid = prediction in valid_responses
            resolved = prediction if is_valid else 'no'

        return is_valid, resolved

    @staticmethod
    def _get_first_word(prediction):
        """
        Extracts the first word, stripping commas, periods, and semicolons. Returns an empty string if an empty string
        is given.
        """
        return str(prediction).strip().split()[0].strip(',:."\'') if prediction else ''

    def _get_score(self, evaluator, predictions, dataset_name):
        metric = self._get_metric(dataset_name)
        # Normalize scores to <= 1, except for BIOSSES, which is already <= 1:
        score = evaluator.evaluate(predictions=predictions, split='test')[metric]
        if not dataset_name == BIOSSES:
            score = score / 100

        return f'{score:.4f}'  # normalize scores to 4 decimals

    def _save_predictions(self, model_name, dataset_name, template_name, predictions):
        csv_name = self._get_scenario_hash(model_name, dataset_name, template_name)
        predictions_path = Path(self.temp_predictions_dir, csv_name).with_suffix('.csv')
        predictions_df = DataFrame(predictions, columns=PREDICTIONS_FILE_HEADERS)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_df.to_csv(predictions_path, index=False)

    def _load_saved_predictions(self, dataset_name: str, template_name: str, model_name: str) -> dict:
        """
        Loads predictions for the given dataset/prompt template combination from the saved aggregated table of stats.

        Args:
            dataset_name: Dataset name.
            template_name: Prompt template name.
            model_name: Model name.

        Returns:
            dict: Predictions dict.
        """
        pred_df = read_csv(self.predictions_file, na_filter=False, dtype=str)
        preds = pred_df[
            (pred_df['Model'] == model_name) &
            (pred_df['Dataset'] == dataset_name) &
            (pred_df['Prompt template'] == template_name)
        ][['Example ID', 'Prediction (resolved)']]

        # Parse predictions for certain datasets to fit the corresponding schema.
        if dataset_name in {BIOSSES, HOC}:
            preds['Prediction (resolved)'] = preds['Prediction (resolved)'].apply(ast.literal_eval)

        predictions = {
            'dataset_name': dataset_name,
            'model_name': model_name,
            'predictions': {example_id: pred for _, example_id, pred in preds.itertuples()}
        }
        return predictions

    def _aggregate_results(self, scores_only: bool = False) -> None:
        """
        Aggregates the individual prompt stats, predictions, and score CSVs and merges them into a single file.

        Args:
            scores_only: If True, only aggregate the score CSVs.

        Returns:
            None
        """
        if not scores_only:
            self._merge_df(self.temp_prompt_stats_dir, self.prompt_stats_file, 'prompt stats')
            self._merge_df(self.temp_predictions_dir, self.predictions_file, 'predictions')
        scores_df = self._merge_df(self.temp_scores_dir, self.scores_file, 'scores')

    def _merge_df(self, in_directory, out_file, name):
        df = None
        for f in Path(in_directory).glob('*.csv'):
            if df is None:
                df = read_csv(f, na_filter=False)
            else:
                df = concat([df, read_csv(f, na_filter=False)], ignore_index=True, sort=False)
        if df is None:
            self.logger.warning(f'No {name} to save')
        else:
            df.to_csv(out_file, index=False)

        return df


if __name__ == '__main__':
    _args = _parse_args()
    if _args.chat_completion:
        config = CHAT_PROMPT_CONFIG
    else:
        config = DEFAULT_PROMPT_CONFIG
    r = Runner(
        config=load_yaml(config),
        models=_args.models,
        output_dir=_args.output_dir,
        example_map_file=_args.example_map_file,
        datasets=_args.datasets or DATASETS,
        limit=_args.limit,
        ip=_args.ip,
        overwrite=_args.overwrite,
        client=_args.client
    )
    r.run(chat_completion_enabled=_args.chat_completion, use_system_role=_args.use_system_role)
