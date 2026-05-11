"""Defines locations within the project structure (for root directories, games and results)
and supplies several functions for loading and storing files."""

from typing import Dict
import os
import json
import csv

######### path construction functions ###################

def project_root():
    """Get the absolute path to main clembench directory.
    Returns:
         The absolute path to main clembench directory as string.
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def clemcore_root():
    """Get the absolute path to the framework directory.
    Returns:
        The absolute path to the framework directory (clembench/framework) as string.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def results_root(results_dir: str) -> str:
    """Get the absolute path to the results root directory.
    Args:
        results_dir: The relative path to the results directory inside the clembench directory.
    """
    if os.path.isabs(results_dir):
        return results_dir
    # if not absolute, results_dir is given relative to project root (see default in cli.py)
    return os.path.normpath(os.path.join(project_root(), results_dir))


def game_results_dir(results_dir: str, dialogue_pair: str, game_name: str):
    """Get the absolute path to the results directory for a specified game and player pair combination.
    Args:
        results_dir: The relative path to the results directory inside the clembench directory.
        dialogue_pair: The name of the player pair combination directory.
        game_name: The name of the game (directory).
    Returns:
        The absolute path to the results directory for a specified game and player pair combination as string.
    """
    return os.path.join(results_root(results_dir), dialogue_pair, game_name)


def file_path(file_name: str, game_path: str = None) -> str:
    """
    Get absolute path to a specific file
    TODO check if this is actually ever called without a game_path
    Args:
        file_name: the path to a file (can be a path relative to the game directory)
        game_path: the path to the game directory (optinal)

    Returns: The absolute path to a file relative to the game directory (if specified) or the clembench directory

    """
    if game_path:
        if os.path.isabs(game_path):
            return os.path.join(game_path, file_name)
        else:
            return os.path.join(project_root(), game_path, file_name)
    return os.path.join(project_root(), file_name)


########### file loading functions #########################


def load_csv(file_name: str, game_path: str):
    """Load a CSV file from a clemgame.
    Args:
        file_name: Name of the CSV file.
        game_name: The name of the game that the CSV file belongs to.
    Returns:
        A list version of the CSV file content.
    """
    # iso8859_2 was required for opening nytcrosswords.csv for clues in wordle
    rows = []
    fp = file_path(file_name, game_path)
    with open(fp, encoding='iso8859_2') as csv_file:
        data = csv.reader(csv_file, delimiter=',')
        # header = next(data)
        for row in data:
            rows.append(row)
    return rows


def load_json(file_name: str, game_path: str) -> Dict:
    """Load a JSON file from a clemgame.
    Args:
        file_name: Name of the JSON file.
        game_name: The name of the game that the JSON file belongs to.
    Returns:
        A dict of the JSON file content.
    """
    data = load_file(file_name, game_path, file_ending=".json")
    data = json.loads(data)
    return data


def load_template(file_name: str, game_path: str) -> str:
    """Load a text template file from a clemgame.
    Args:
        file_name: Name of the text template file.
        game_name: The name of the game that the text template file belongs to.
    Returns:
        A string version of the text template file content.
    """
    # TODO this a bit redundant and could be removed by changing all usages
    #  of load_template (and GameResourceLocator.load_template()) to directly use load_file(..., file_ending=".template")
    return load_file(file_name, game_path, file_ending=".template")


def load_file(file_name: str, game_path: str = None, file_ending: str = None) -> str:
    """Load a file from a clemgame.
    Assumes the file to be an utf8-encoded (text) file.
    Args:
        file_name: Name of the file.
        game_name: The name of the game that the file belongs to.
        file_ending: The file type suffix of the file.
    Returns:
        The file content as returned by open->read().
    """
    if file_ending and not file_name.endswith(file_ending):
        file_name = file_name + file_ending
    fp = file_path(file_name, game_path)
    with open(fp, encoding='utf8') as f:
        data = f.read()
    return data


def load_results_json(file_name: str, results_dir: str, dialogue_pair: str, game_name: str) -> Dict:
    file_ending = ".json"
    if not file_name.endswith(file_ending):
        file_name = file_name + file_ending
    fp = os.path.join(game_results_dir(results_dir, dialogue_pair, game_name), file_name)
    with open(fp, encoding='utf8') as f:
        data = f.read()
    data = json.loads(data)
    return data

########### file storing function ################


def store_file(data, file_name: str, dir_path: str, sub_dir: str = None, do_overwrite: bool = True) -> str:
    """Store a file.
    Base function to handle relative clembench directory paths.
    Args:
        data: Content to store in the file.
        file_name: Name of the file to store.
        dir_path: Path to the directory to store the file to.
        sub_dir: Optional subdirectories to store the file in.
        do_overwrite: Determines if existing file should be overwritten. Default: True
    Returns:
        The path to the stored file.
    """
    if sub_dir:
        dir_path = os.path.join(dir_path, sub_dir)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    fp = os.path.join(dir_path, file_name)
    if not do_overwrite:
        if os.path.exists(fp):
            raise FileExistsError(fp)

    with open(fp, "w", encoding='utf-8') as f:
        if file_name.endswith(".json"):
            json.dump(data, f, ensure_ascii=False)
        else:
            f.write(data)
    return fp
