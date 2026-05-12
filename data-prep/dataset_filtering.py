import gzip
import json
from clemcore.clemgame import GameRegistry
from datasets import load_dataset


dataset = load_dataset(
    "colab-potsdam/playpen-data",
    "interactions",
    split="train")


def filter_failures(dataset):
    return [round for round in dataset
            if round["meta"]["outcome"] in ["failure", "aborted"]]


def inspect_failures(failures_data, n=10):
    """A function to inspect the retained failed rounds by selecting a specific number of those to print to console.
    count is set to 10 be default, can be modified as needed.
    """
    print(f"Total failures: {len(failures_data)}")
    print(f"\nFirst {n} records:")
    print(json.dumps(failures_data[:n], indent=2))


def save_failures(failures_data, create_json=True):
    """A function to save the failed rounds in the playpen dataset to a gzip file for memory efficiency.
    If create_json=True, also write a JSON file for readability
    """
    with gzip.open("failures_data.json.gz", 'wt', encoding='utf-8') as f:
        json.dump(failures_data, f, indent=2)

    if create_json:
        with open("failures_data.json", 'w', encoding='utf-8') as f:
            json.dump(failures_data, f, indent=2)


def list_clem_games():
    '''Function to list all the names of games from clembench. Will be moved to reasoning traces generation inference pipeline
    '''
    game_registry = GameRegistry.from_directories_and_cwd_files()
    all_games = game_registry.get_name_specs()
    for game in all_games:
        game["game_name"]


def split_by_game(failures_data):
    '''Incomplete function to split the dataset at specific game names. Will be either completed or deleted if deemed unnecessary
    '''
    with open(failures_data, 'r') as f:
        failures = failures_data
        game_names = list_clem_games()


def extract_failures(dataset, create_json=True, inspect_only=False, n=10):
    """Convenience wrapper for failure extraction"""
    failures = filter_failures(dataset)

    if inspect_only:
        inspect_failures(failures, n=n)
    else:
        save_failures(failures, create_json=create_json)

    return failures


def main():
    print(
        extract_failures(
            dataset,
            create_json=True,
            inspect_only=False,
            n=10))


if __name__ == "__main__":
    main()
