import gzip
import json
from datasets import load_dataset


dataset = load_dataset(
    "colab-potsdam/playpen-data",
    "interactions",
    split="train")


def filter_failures(dataset):
    """A function to load game rounds (conversations) from the playpen data HF dataset and filter out successful rounds.
    """
    for round in dataset:
        if round["meta"]["outcome"] == "failure" or round["meta"]["outcome"] == "aborted":
            return round


def inspect_failures(data, n=10):
    """A function to inspect the retained failed rounds by selecting a specific number of those to print to console.
    count is set to 10 be default, can be modified as needed.
    """
    print(f"Total failures: {len(data)}")
    print(f"\nFirst {n} records:")
    print(json.dumps(data[:n], indent=2))


def save_failures(failures_data, create_json=True):
    """A function to save the failed rounds in the playpen dataset to a gzip file for memory efficiency.
    If create_json=True, also write a JSON file for readability
    """
    with gzip.open("failures_data.json.gz", 'wt', encoding='utf-8') as f:
        json.dump(failures_data, f, indent=2)

    if create_json:
        with open("failures_data.json", 'w', encoding='utf-8') as f:
            json.dump(failures_data, f, indent=2)


def extract_failures(dataset, create_json=True, inspect_only=False, n=10):
    """Convenience wrapper for failure extraction"""
    failures = filter_failures(dataset)

    if inspect_only:
        inspect_failures(failures, n=n)
    else:
        save_failures(failures, create_json=create_json)

    return failures

def main():
    print(extract_failures(dataset, create_jason=True, inspect_only=False, n=10))

if __name__ == "__main__":
    main()
