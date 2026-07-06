"""Script to run DPO fine-tuning starting from the merged SFT checkpoint
produced by sft.py then merge_adapters.py.
Can be used to run multiple training ablations based on the preference pairs files passed via command line.
"""

import json
import random
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

_HERE = Path(__file__).parent


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
        description="DPO fine-tuning on top of merged SFT fine-tuned checkpoint.")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the merged SFT checkpoint produced by sft.py then merge_adapters.py.",
    )
    parser.add_argument(
        "--pairs_files",
        nargs="+",
        required=True,
        help="Selects the ready-made DPO pairs files to use for training, depending on the ablation of interest.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap on the number of training entries.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Quantize to 4-bit (nf4).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Where to save the DPO checkpoints, name it per ablation to avoid overwriting."
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="DPO beta (KL anchor strength).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Model: {args.model}, samples: {args.max_samples or 'all'}, "
          f"epochs: {args.epochs}, output: {args.output_dir}")

    pairs = []
    for p in args.pairs_files:
        with open(p) as f:
            part = json.load(f)
        print(f"  {Path(p).name}: {len(part)} pairs")
        pairs.extend(part)
    pairs = [{"prompt": t["prompt"], "chosen": t["chosen"],
              "rejected": t["rejected"]} for t in pairs]
    random.seed(args.seed)
    random.shuffle(pairs)
    if args.max_samples is not None:
        pairs = pairs[:args.max_samples]
    split = int(0.9 * len(pairs))
    train_ds = Dataset.from_list(pairs[:split])
    eval_ds = Dataset.from_list(pairs[split:])

    print(f"DPO pairs - train: {len(train_ds)}, eval: {len(eval_ds)}\n")

    model, tokenizer = load_model_and_tokenizer(
        args.model, use_4bit=args.use_4bit)

    cfg = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
        learning_rate=5e-5,
        bf16=True,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        seed=args.seed,
        beta=args.beta,
        max_length=2048,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=make_lora_config(r=args.lora_r),
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Checkpoint saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
