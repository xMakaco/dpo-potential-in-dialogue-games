"""Script to generate reflection comments on on-policy rollouts collected by
collect_onpolicy_pairs.py. The script imports llm_wrapper.py and prompts.py
to call an LLM to generate such reflection comments. The comments are appended as
trailing assistant messages. Use --condition to choose whether to generate comments
for aborted or failed rounds, the right prompt will be used accordingly.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from llm_wrapper import GptAzureClient, DeepseekClient
from prompts import build_prompt

_HERE = Path(__file__).parent
OUT_DIR = _HERE.parent / "data" / "dpo_pairs"

OUTCOME = {"aborted": "aborted", "failed": "failed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        required=True,
        choices=[
            "aborted",
            "failed"])
    parser.add_argument(
        "--input",
        required=True,
        help="onpolicy_aborted/failed_rounds.*.json from collect_onpolicy_pairs.py")
    parser.add_argument(
        "--model_id",
        required=True,
        help="judge model id, e.g. gpt-5.2")
    parser.add_argument(
        "--backend",
        choices=[
            "azure",
            "openai"],
        default="azure")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    outcome = OUTCOME[args.condition]
    model = GptAzureClient(
        args.model_id) if args.backend == "azure" else DeepseekClient(
        args.model_id)

    in_path = Path(args.input)
    OUT_DIR.mkdir(exist_ok=True)
    stem = in_path.stem
    for p in ("onpolicy_aborted.", "onpolicy_failed_rounds."):
        stem = stem.replace(p, "")
    out_path = OUT_DIR / \
        f"onpolicy_commented.{args.condition}.{stem}.{args.model_id}.json"
    error_log_path = str(out_path) + ".errors.log"

    if out_path.exists() and not args.resume:
        sys.exit(
            f"{out_path} exists. Use --resume to continue it. Refusing to overwrite.")

    records = json.load(open(in_path))

    existing = []
    if args.resume and out_path.exists():
        existing = json.load(open(out_path))
        done = {r["pair_id"] for r in existing}
        records = [r for r in records if r["pair_id"] not in done]
        print(f"Resuming: {len(done)} done, {len(records)} remaining.")

    def process(rec):
        clean = {"messages": rec["messages"], "meta": rec["meta"]}
        prompt_messages = build_prompt(clean, outcome=outcome)
        if prompt_messages is None:
            raise ValueError(
                f"build_prompt returned None (outcome mismatch: {rec['meta'].get('outcome')})")
        comment = model.generate_comment(prompt_messages)
        if not comment or not comment.strip():
            raise ValueError("empty comment")
        out = dict(rec)
        out["comment"] = comment.strip()
        out["messages"] = rec["messages"] + \
            [{"role": "assistant", "content": comment.strip()}]
        return out

    processed = []
    with open(error_log_path, "w") as error_log:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process, r) for r in records]
            try:
                for future in as_completed(futures):
                    try:
                        processed.append(future.result())
                        print(f"Done: {len(processed)}/{len(records)}")
                    except Exception as e:
                        error_log.write(f"{e}\n")
                        error_log.flush()
            except KeyboardInterrupt:
                print(f"\nInterrupted — saving {len(processed)} completed...")
                executor.shutdown(wait=False, cancel_futures=True)

    with open(out_path, "w") as f:
        json.dump(existing + processed, f, indent=2)
    print(
        f"Saved {len(existing) + len(processed)} commented rounds to {out_path}")


if __name__ == "__main__":
    main()
