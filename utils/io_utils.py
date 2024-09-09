import inspect
import json
from pathlib import Path
from typing import Union, Callable

import yaml


def load_yaml(yaml_file: Union[str, Path]) -> dict:
    """
    Loads the content of a YAML file.
    """
    with open(yaml_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise exc


def save_json(data: Union[dict, list], fp: Union[str, Path], parents: bool = False, **kwargs) -> None:
    """
    Saves dictionary to a designated filepath.

    Args:
        data: dictionary
        fp: path to the file to save the contents of the dictionary to
        parents: whether to make existing parent directories

    Returns:
        None
    """
    if parents:
        make_parent_dirs(fp)
    with open(fp, 'w', encoding='utf-8') as f:
        json.dump(data, f, **kwargs)


def load_json(fp: Union[str, Path]) -> dict:
    """
    Loads contents of json file.

    Args:
        fp:  filepath to json file

    Returns:
        dict: Contents of the json file as a dictionary
    """
    with open(fp, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_txt(txt: str, fp: Union[str, Path], parents: bool = False) -> None:
    """
    Saves text to a designated filepath.

    Args:
        txt: The text
        fp: path to the file to save the text to
        parents: whether to make existing parent directories

    Returns:
        None
    """
    if parents:
        make_parent_dirs(fp)
    with open(fp, 'w', encoding='utf-8') as f:
        f.write(txt)


def load_txt(fp: Union[str, Path]) -> str:
    """
    Loads contents of a text file.

    Args:
        fp:  filepath to text file

    Returns:
        str: Contents of the text file as a str
    """
    with open(fp, 'r', encoding='utf-8') as f:
        return f.read()


def json_to_str(json_obj: dict) -> str:
    return json.dumps(json_obj, indent=2)


def make_dir(directory: Union[str, Path]) -> Path:
    """Makes a directory relative to the current working directory"""
    directory = Path(directory)
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def default_config():
    parent = Path(__file__).parent
    config = load_yaml(parent / 'base_config.yml')
    # Replace relative path to LRE config to absolute:
    config['lre_config_path'] = str(parent / config['lre_config_path'])

    return config


def filter_args(args: dict, func: Callable) -> dict:
    """Filters out from a dictionary of arguments, any arguments not required by a given function."""
    return {k: args[k] for k in args.keys() if k in inspect.getfullargspec(func).args}


def make_parent_dirs(file_path: Union[str, Path]) -> None:
    if isinstance(file_path, str):
        file_path = Path(file_path)
    file_path.parent.mkdir(exist_ok=True, parents=True)
