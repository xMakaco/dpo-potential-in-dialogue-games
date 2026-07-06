"""Script to conduct SFT on successful game rounds from the playpen dataset (outcome == success),
preparing a model for subsequent DPO training.
"""

import json
import random
import argparse
from pathlib import Path

import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

_HERE = Path(__file__).parent

HF_DATASET = "colab-potsdam/playpen-data"
HF_CONFIG = "interactions"
HF_SPLIT = "train"


def load_success_rounds(
    samples_per_game=None,
    seed=42,
):
    """
    Load successful game rounds from the playpen HuggingFace dataset (outcome == success).
    Metadata is stripped.
    If samples_per_game is set, at most that many examples are drawn from
    each game type.
    """
    print(f"Loading {HF_DATASET} ({HF_CONFIG}/{HF_SPLIT})...")
    hf_data = load_dataset(HF_DATASET, HF_CONFIG, split=HF_SPLIT)

    rng = random.Random(seed)

    if samples_per_game is not None:
        # stratified sampling: at most samples_per_game from each game type, so
        # all games are equally represented if training samples are capped.
        by_game = {}
        for entry in hf_data:
            if entry["meta"]["outcome"] != "success":
                continue
            game = entry["meta"]["game"]
            by_game.setdefault(game, []).append(
                {"messages": entry["messages"]})

        entries = []
        for game, rounds in sorted(by_game.items()):
            rng.shuffle(rounds)
            sampled = rounds[:samples_per_game]
            entries.extend(sampled)
            print(f"  {game}: {len(sampled)}/{len(rounds)} rounds")

        rng.shuffle(entries)
        print(
            f"Stratified sample: {len(entries)} rounds across {len(by_game)} games.")
    else:
        entries = [
            {"messages": entry["messages"]}
            for entry in hf_data
            if entry["meta"]["outcome"] == "success"
        ]
        print(f"Found {len(entries)} successful rounds.")
        rng.shuffle(entries)

    return Dataset.from_list(entries)


def load_model_and_tokenizer(model_id, use_4bit=False):
    """
    Load the model in 4-bit (nf4) or bfloat16, plus its tokenizer.
    """
    if use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_cfg,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def make_lora_config(r=16):
    return LoraConfig(
        r=r,
        lora_alpha=r * 2,
        target_modules="all-linear",
        lora_dropout=0.05,
        modules_to_save=[],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def main():
    parser = argparse.ArgumentParser(
        description="SFT on successful clembench game rounds."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-2B",
        help="HuggingFace model ID or path to a local checkpoint.",
    )
    parser.add_argument(
        "--samples_per_game",
        type=int,
        default=None,
        help="If set, sample at most this many examples per game type. "
             "Ensures all game formats are represented when capping training data.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Quantize to 4-bit (nf4).",
    )
    parser.add_argument(
        "--grad_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing."
    )
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument(
        "--data_file",
        default=None,
        help="Optional JSON file of [{'messages': [...]}, ...] to train on instead of the "
             "playpen success rounds. Useful for the chosen-only ablation.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Defaults to ./checkpoints/sft/<model>.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output_dir is None:
        model_slug = args.model.split("/")[-1]
        args.output_dir = str(
            _HERE /
            "checkpoints" /
            "sft" /
            f"{model_slug}-all-linear")

    print(
        f"Model: {args.model}, samples per game: {args.samples_per_game or 'all'}, "
        f"epochs: {args.epochs}, output: {args.output_dir}")

    model, tokenizer = load_model_and_tokenizer(
        args.model, use_4bit=args.use_4bit)

    if args.data_file:
        entries = json.load(open(args.data_file))
        dataset = Dataset.from_list(
            [{"messages": e["messages"]} for e in entries])
        print(f"Loaded {len(dataset)} examples from {args.data_file}")
    else:
        dataset = load_success_rounds(
            samples_per_game=args.samples_per_game,
            seed=args.seed,
        )
    split = dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"SFT dataset - train: {len(train_ds)}, eval: {len(eval_ds)}\n")

    cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=args.grad_checkpointing,
        gradient_checkpointing_kwargs={
            "use_reentrant": False} if args.grad_checkpointing else None,
        warmup_ratio=0.05,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        seed=args.seed,
        report_to="none",
        completion_only_loss=True,
        max_length=1024,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=make_lora_config(r=args.lora_r),
    )

    print("Starting SFT...")
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"SFT checkpoint saved to: {args.output_dir}")
    print(f"To run DPO: python dpo.py --model {args.output_dir}")


if __name__ == "__main__":
    main()
