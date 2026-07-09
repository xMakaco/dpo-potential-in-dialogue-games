"""Script to distill the reflection-commented rounds from generate_onpolicy_comments.py
into ready-to-train DPO triples. The LLM judge is given the rounds and the existing
reflection comment and asked only for the concrete corrected move.
For failed rounds it also returns the move_index of the flagged move, so rejected is exactly the move the
comment refers to. Rounds with no genuine correction (invalid index, empty move, or
corrected == original) are skipped. chosen is the bare corrected move with no prose.
This choice was made after multiple observations that textual reflections bleed into the model's output style,
leading it to create even more verbose outputs than before, causing a significant drop in performance.
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from llm_wrapper import GptAzureClient, DeepseekClient

_HERE = Path(__file__).parent
OUT_DIR = _HERE.parent / "data" / "dpo_pairs"

ABORTED_DISTILL = (
    "You are given a dialogue game round that was rejected by the game master, the player's "
    "rejected move, and an expert reflection explaining what was wrong. Extract the concrete fix.\n\n"
    'Return a JSON object with one field: "corrected_move" - a single replacement response in the '
    "game's EXACT required format, legal in that game state, consistent with the reflection. "
    "ONLY the move, no commentary, no extra lines. Answer with the JSON object only.")

FAILED_DISTILL = (
    "You are given the transcript of a dialogue game round the player FAILED, a numbered list of "
    "the player's own moves, and an expert reflection on what was played suboptimally.\n\n"
    "Using the reflection, return a JSON object with two fields:\n"
    '  "move_index": the integer index (from the numbered list) of the first move the reflection '
    "identifies as suboptimal.\n"
    '  "corrected_move": a single better replacement move in the game\'s EXACT required format, '
    "legal in that game state. ONLY the move, no commentary. Answer with the JSON object only.")


def _json_obj(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    return json.loads(m.group(0))


def context_before_move(messages, move_index):
    """Messages up to (not including) the move_index-th assistant turn."""
    seen, out = 0, []
    for m in messages:
        if m["role"] == "assistant":
            if seen == move_index:
                return out
            seen += 1
        out.append(m)
    return out


def build_aborted_msgs(rec):
    # rec["messages"] = context + [violating move] + [reflection comment]
    convo = json.dumps(rec["messages"][:-2], indent=2)
    user = (
        f"CONVERSATION (game master 'user', player 'assistant'):\n{convo}\n\n"
        f"PLAYER'S REJECTED MOVE:\n{rec['rejected']}\n\n"
        f"GAME MASTER'S ERROR:\n{rec.get('gm_error','')}\n\n"
        f"EXPERT REFLECTION:\n{rec['comment']}"
    )
    return [{"role": "system", "content": ABORTED_DISTILL},
            {"role": "user", "content": user}]


def build_failed_msgs(rec):
    transcript = json.dumps(rec["messages"][:-1],
                            indent=2)  # drop trailing comment
    numbered = "\n".join(f"[{i}] {m}" for i, m in enumerate(rec["moves"]))
    user = (
        f"ROUND TRANSCRIPT (game master 'user', player 'assistant'):\n{transcript}\n\n"
        f"THE PLAYER'S MOVES (numbered):\n{numbered}\n\n"
        f"EXPERT REFLECTION:\n{rec['comment']}")
    return [{"role": "system", "content": FAILED_DISTILL},
            {"role": "user", "content": user}]


def make_aborted_pair(rec, corrected):
    rejected = rec["rejected"]
    if not corrected or corrected.strip() == str(rejected).strip():
        return None
    return {
        "pair_id": rec["pair_id"], "game": rec["meta"]["game"],
        "prompt": rec["messages"][:-2],
        "chosen": [{"role": "assistant", "content": corrected}],
        "rejected": [{"role": "assistant", "content": rejected}],
    }


def make_failed_pair(rec, move_index, corrected):
    moves = rec["moves"]
    if not (0 <= move_index < len(moves)):
        return None
    rejected = moves[move_index]
    if not corrected or corrected.strip() == str(rejected).strip():
        return None
    return {
        "pair_id": rec["pair_id"], "game": rec["meta"]["game"],
        "prompt": context_before_move(rec["messages"][:-1], move_index),
        "chosen": [{"role": "assistant", "content": corrected}],
        "rejected": [{"role": "assistant", "content": rejected}],
    }


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
        help="onpolicy_commented.<condition>.*.json")
    parser.add_argument("--model_id", required=True)
    parser.add_argument(
        "--backend",
        choices=[
            "azure",
            "openai"],
        default="azure")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    model = GptAzureClient(
        args.model_id) if args.backend == "azure" else DeepseekClient(
        args.model_id)

    in_path = Path(args.input)
    OUT_DIR.mkdir(exist_ok=True)
    stem = in_path.stem
    for p in (f"onpolicy_commented.{args.condition}.", ):
        stem = stem.replace(p, "")
    out_path = OUT_DIR / f"onpolicy_pairs.{args.condition}.{stem}.json"
    error_log_path = str(out_path) + ".errors.log"

    if out_path.exists() and not args.resume:
        sys.exit(
            f"{out_path} exists. Use --resume to continue it. Refusing to overwrite.")

    records = json.load(open(in_path))

    existing = []
    if args.resume and out_path.exists():
        existing = json.load(open(out_path))
        done = {t["pair_id"] for t in existing}
        records = [r for r in records if r["pair_id"] not in done]
        print(f"Resuming: {len(done)} done, {len(records)} remaining.")

    def process(rec):
        if args.condition == "aborted":
            obj = _json_obj(
                model.generate_comment(
                    build_aborted_msgs(rec),
                    max_completion_tokens=512))
            pair = make_aborted_pair(
                rec, str(obj.get("corrected_move", "")).strip())
        else:
            obj = _json_obj(
                model.generate_comment(
                    build_failed_msgs(rec),
                    max_completion_tokens=512))
            pair = make_failed_pair(
                rec, int(
                    obj["move_index"]), str(
                    obj.get(
                        "corrected_move", "")).strip())
        if pair is None:
            raise ValueError("skip: no genuine aligned correction")
        return pair

    processed, skipped = [], 0
    with open(error_log_path, "w") as error_log:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process, r) for r in records]
            try:
                for future in as_completed(futures):
                    try:
                        processed.append(future.result())
                        print(f"Done: {len(processed)}/{len(records)}")
                    except Exception as e:
                        if str(e).startswith("skip:"):
                            skipped += 1
                        error_log.write(f"{e}\n")
                        error_log.flush()
            except KeyboardInterrupt:
                print(f"\nInterrupted - saving {len(processed)} pairs...")
                executor.shutdown(wait=False, cancel_futures=True)

    with open(out_path, "w") as f:
        json.dump(existing + processed, f, indent=2)
    print(
        f"Saved {len(existing) + len(processed)} pairs to {out_path}  (skipped {skipped})")


if __name__ == "__main__":
    main()
