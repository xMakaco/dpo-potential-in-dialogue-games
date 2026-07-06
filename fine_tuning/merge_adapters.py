"""Script to merge LoRA SFT and DPO adapters into either a base model or the merged model from the previous fine-tuning stage.
Since small models of the Qwen/Qwen3.5 family ship without a generation_config.json file, this script also ensures that the merged model has a correct generation_config.json with an appropriate eos_token_id and pad_token_id,
in order to avoid the model hallucinating extra conversation turns after the end of a chat turn, and breaking clemcore's automatic EOS culling.
If a model which ships with a generation_config.json file is used, the script will trust that file and preserve its values.
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def infer_base_model(adapter_dir):
    """Read the base model id from the adapter's adapter_config.json."""
    with open(adapter_dir / "adapter_config.json") as f:
        return json.load(f)["base_model_name_or_path"]


def resolve_token_id(tokenizer, token):
    """Return the vocab id for `token`, or None if it is not a real token.
    convert_tokens_to_ids returns the unk id for unknown tokens, which must not be added to the EOS list.
    """
    tid = tokenizer.convert_tokens_to_ids(token)
    if tid is None:
        return None
    unk = getattr(tokenizer, "unk_token_id", None)
    if tid == unk and token != getattr(tokenizer, "unk_token", None):
        return None
    return tid


def build_generation_config(base_id, tokenizer):
    """Build a correct GenerationConfig for any base model.
    Returns (generation_config, source_description).
    It trusts the base model's shipped generation config when it declares EOS.
    """
    try:
        gen_cfg = GenerationConfig.from_pretrained(base_id)
    except Exception:
        gen_cfg = None
    if gen_cfg is not None and gen_cfg.eos_token_id is not None:
        if gen_cfg.pad_token_id is None:
            gen_cfg.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        return gen_cfg, "base model's generation_config.json"

    eos_ids = []
    for tok in ("<|im_end|>", "<|endoftext|>"):
        tid = resolve_token_id(tokenizer, tok)
        if tid is not None and tid not in eos_ids:
            eos_ids.append(tid)
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in eos_ids:
        eos_ids.append(tokenizer.eos_token_id)
    if not eos_ids:
        raise SystemExit(
            f"Could not resolve any EOS token id for {base_id}; "
            "no shipped generation_config and the tokenizer declares no EOS."
        )
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_ids[-1]
    return GenerationConfig(
        eos_token_id=eos_ids, pad_token_id=pad_id), "reconstructed from tokenizer"


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model.")
    parser.add_argument(
        "--adapter",
        required=True,
        help="Path to the LoRA adapter checkpoint.")
    parser.add_argument(
        "--base",
        default=None,
        help="Base model HF id. Defaults to base_model_name_or_path from adapter_config.json.")
    parser.add_argument("--out", default=None,
                        help="Output directory. Defaults to <adapter>-merged.")
    args = parser.parse_args()

    adapter_dir = Path(args.adapter).resolve()
    base_id = args.base or infer_base_model(adapter_dir)
    if args.out:
        out_dir = Path(args.out).resolve()
    else:
        out_dir = adapter_dir.parent / f"{adapter_dir.name}-merged"

    print(f"Base: {base_id}")
    print(f"Adapter: {adapter_dir}")
    print(f"Output: {out_dir}\n")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            adapter_dir, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            base_id, trust_remote_code=True)

    gen_cfg, source = build_generation_config(base_id, tokenizer)
    print(f"Generation config ({source}):")
    print(f"  eos_token_id: {gen_cfg.eos_token_id}")
    print(f"  pad_token_id: {gen_cfg.pad_token_id}\n")

    print("Loading base model (CPU, bf16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print("Merging adapter...")
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    merged = peft_model.merge_and_unload()

    merged.generation_config = gen_cfg

    print(f"Saving to {out_dir}...")
    merged.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    with open(out_dir / "generation_config.json") as f:
        written = json.load(f)
    print("\ngeneration_config.json as written:")
    print(json.dumps(written, indent=2))
    eos = written.get("eos_token_id")
    assert eos, f"No eos_token_id in saved config: {eos!r}"
    print("Merge complete and generation config verified.")


if __name__ == "__main__":
    main()
