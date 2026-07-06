"""Smoke test: is the native <think> channel still alive after SFT?

Replays a real clembench first-turn prompt (wordle, taken verbatim from an
on-policy rollout log) against a checkpoint twice: once exactly as past evals
rendered it (empty <think>\n\n</think> block = thinking off) and once with an
open <think>\n (thinking on). Prints both raw completions plus diagnostics:
whether the think block is closed, its length, and whether a parser-valid
move follows it.

Usage:
    python think_smoke_test.py --model checkpoints/sft/Qwen3.5-9B-all-linear-merged
    python think_smoke_test.py --model Qwen/Qwen3.5-9B   # base, for comparison
"""

import json
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = Path(__file__).parent
ROLLOUT = (_HERE.parent / "onpolicy-rollouts/Qwen3.5-2B-sft-full-merged/wordle/"
           "medium_frequency_words_no_clue_no_critic/instance_00004/player_1.requests.json")
THINK_OFF_TAIL = "<think>\n\n</think>\n\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--with_off", action="store_true",
                        help="also run the think-off control (default: think-on only, for speed)")
    args = parser.parse_args()

    rec = json.load(open(ROLLOUT))
    prompt_off = rec["calls"][0]["call"]["manipulated_prompt_obj"]["inputs"]
    assert prompt_off.endswith(THINK_OFF_TAIL), "rollout prompt tail changed; update THINK_OFF_TAIL"
    prompt_on = prompt_off[: -len(THINK_OFF_TAIL)] + "<think>\n"

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    variants = [("THINK ON", prompt_on)]
    if args.with_off:
        variants.insert(0, ("THINK OFF (as in past evals)", prompt_off))
    for label, prompt in variants:
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)

        print(f"\n{'=' * 20} {label} {'=' * 20}")
        print(completion)
        print("-" * 60)
        if "THINK ON" in label:
            closed = "</think>" in completion
            print(f"think block closed: {closed}")
            if closed:
                think, after = completion.split("</think>", 1)
                print(f"think length: {len(think.strip())} chars")
                print(f"post-think text starts with: {after.strip()[:120]!r}")
                print(f"post-think contains 'guess:': {'guess:' in after}")
            else:
                print("!! no </think> within budget — channel damaged or budget too small")


if __name__ == "__main__":
    main()
