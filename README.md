# Reflections-Augmented SFT for Dialogue Games

SFT + DPO post-training of Qwen3.5 models for the clembench / LM Playpen dialogue-game
benchmark. The headline result: a two-stage pipeline (SFT on successful game rounds,
then DPO on content-controlled "anti-bleed" preference pairs) takes Qwen3.5-9B from
58.82 to **70.21 clemscore** — and the ablations show *where* that gain actually comes
from. DPO reliably reinforces behaviour the model already has (clean, parseable moves);
it does not teach new strategy. The further the preference target moves off-policy
(GPT-corrected moves for aborted or lost rounds), the worse the result gets.

**Final model:** [`Makaco/lmps-challenge-qwen3.5-9b-dpo`](https://huggingface.co/Makaco/lmps-challenge-qwen3.5-9b-dpo)
**DPO pairs dataset:** [`Makaco/playpen-antibleed-dpo-qwen3.5-9b`](https://huggingface.co/datasets/Makaco/playpen-antibleed-dpo-qwen3.5-9b)
**Full training details:** [TRAINING_CARD.md](TRAINING_CARD.md)

The name of the repo is a leftover from the original idea (augmenting SFT data with
reflection comments); the project evolved into the preference-pair study described here.

## What's in the repo

```
fine_tuning/              training pipeline
  sft.py                  LoRA SFT on successful playpen rounds (assistant-only loss)
  dpo.py                  LoRA DPO on ready-made preference pairs (--pairs_files)
  merge_adapters.py       merge any adapter into its base; fixes Qwen3.5 generation_config
dpo_pairs_construction/   building the preference pairs
  collect_onpolicy_pairs.py       harvest rounds from scored rollouts (by outcome)
  generate_onpolicy_comments.py   GPT reflection comments on aborted/failed rounds
  distill_onpolicy_pairs.py       comments -> concrete (chosen, rejected) pairs
  prompts.py                      the reflection prompts
  llm_wrapper.py                  Azure OpenAI / DeepSeek clients
data/dpo_pairs/           the exact pair files used for every reported run
eval_results/             score summaries (clemscore/statscore) for every reported run
model_registry.json       clem model configs for all models in the study
game_registry.json        points clem at ../clembench
TRAINING_CARD.md          full hyperparameters, data, compute budget
```

## Setup

```bash
# clembench (the games) must sit NEXT TO this repo — game_registry.json expects ../clembench
git clone https://github.com/clp-research/clembench.git
git clone https://github.com/xMakaco/Reflections-Augmented-SFT-for-Dialogue-Games.git
cd Reflections-Augmented-SFT-for-Dialogue-Games

python3.10 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# only needed for the static-benchmark evals (statscore):
pip install git+https://github.com/lm-playpen/playpen.git
```

Everything below was run on a single H100 in bf16; total training budget for the
best model is ~4.3 GPU-hours (see the training card).

## Reproducing the best model

No API keys needed — the anti-bleed path is fully self-contained (SFT data comes from
the HF Hub, the DPO pairs are in this repo). Defaults in the scripts are the exact
settings of the released model.

```bash
cd fine_tuning

# 1. SFT on successful playpen rounds
python sft.py --model Qwen/Qwen3.5-9B --grad_checkpointing \
  --output_dir checkpoints/sft/Qwen3.5-9B-all-linear

# 2. merge (also bakes in a correct generation_config — see note below)
python merge_adapters.py --adapter checkpoints/sft/Qwen3.5-9B-all-linear

# 3. anti-bleed DPO from the merged SFT model
python dpo.py --model checkpoints/sft/Qwen3.5-9B-all-linear-merged \
  --pairs_files ../data/dpo_pairs/onpolicy_antibleed.Qwen3.5-9B-sft-full-merged.json \
  --output_dir checkpoints/dpo/Qwen3.5-9B-antibleed-only

# 4. merge again -> final model
python merge_adapters.py --adapter checkpoints/dpo/Qwen3.5-9B-antibleed-only
```

Or skip all of it and pull the released checkpoint from HF.

## The ablations

The pair types are the ablation axis. Every reported condition is just a choice of
`--pairs_files`; the trainer is identical.

| pairs | what they contain | on-policy? |
|---|---|---|
| `onpolicy_antibleed.*` | chosen = the SFT model's own winning move; rejected = the *same* move with an appended justification | fully |
| `onpolicy_pairs.aborted.*` | chosen = GPT-corrected move fixing a rule violation; rejected = the violating move | rejected is, chosen isn't |
| `onpolicy_pairs.failed.*` | chosen = GPT's strategically better move in a lost round; rejected = the actual move (`hidden.clean` = hidden-state games with a hindsight-guarded prompt, `verifiable` = all other games — pass both for the full "failed" condition) | rejected is, chosen isn't |
| `chosen_only.*` | the anti-bleed chosen moves alone, as SFT data (via `sft.py --data_file`) — isolates the imitation component of the DPO gain | fully |

Combinations = multiple `--pairs_files`. Example, aborted + failed:

```bash
python dpo.py --model checkpoints/sft/Qwen3.5-9B-all-linear-merged \
  --pairs_files ../data/dpo_pairs/onpolicy_pairs.aborted.Qwen3.5-9B-sft-full-merged.gpt-5.2.json \
                ../data/dpo_pairs/onpolicy_pairs.failed.Qwen3.5-9B-sft-full-merged.hidden.clean.json \
                ../data/dpo_pairs/onpolicy_pairs.failed.Qwen3.5-9B-sft-full-merged.verifiable.gpt-5.2.json \
  --output_dir checkpoints/dpo/Qwen3.5-9B-aborted-failed
```

Regenerating the pairs from scratch (instead of using `data/dpo_pairs/`) requires
rollouts (`clem run` + `clem score`) and, for the correction conditions, an Azure
OpenAI or DeepSeek key for the GPT passes:

```bash
cd dpo_pairs_construction
python collect_onpolicy_pairs.py --rollouts <scored-rollout-dir> --condition success   # anti-bleed
python collect_onpolicy_pairs.py --rollouts <scored-rollout-dir> --condition aborted
python generate_onpolicy_comments.py --condition aborted --input ../data/dpo_pairs/onpolicy_aborted.<model>.json --model_id <judge>
python distill_onpolicy_pairs.py     --condition aborted --input ../data/dpo_pairs/onpolicy_commented.aborted.<model>.<judge>.json --model_id <judge>
```

Evaluation instances are excluded from every harvest (`--exclude_eval`) — a
contaminated variant inflated clemscore by ~9 points, so this matters.

## Evaluation

```bash
clem run -g "{'benchmark':['2.0']}" -m <model_name>   # model_name from model_registry.json
clem score
```

Only the released model resolves from the HF Hub directly; all other registry entries
point to `fine_tuning/checkpoints/...` and expect checkpoints produced by the pipeline
above. Score summaries for every reported run are in `eval_results/`
(clemscore = (%played / 100) × quality, on the clembench 2.0 instances).

## Results so far

| model | clemscore |
|---|---|
| Qwen3.5-9B SFT | 58.82 |
| **Qwen3.5-9B SFT + anti-bleed DPO** | **70.21** |
| Qwen3.5-9B SFT + aborted-corrections DPO | 60.96 |
| Qwen3.5-9B chosen-only SFT | 65.36 |
| Qwen3.5-2B SFT + anti-bleed DPO | 44.69 |

Full per-game breakdowns in `eval_results/`; the complete ablation grid (failed
corrections and all combinations, plus the 2B scale point) is being finalised for the
technical report.

## The Qwen3.5 generation_config pitfall

Small Qwen3.5 releases ship without a `generation_config.json` and declare only
`<|endoftext|>` as EOS, so after fine-tuning the model never stops at the chat-turn
boundary (`<|im_end|>`) and hallucinates extra conversation turns.
`merge_adapters.py` reconstructs a correct config (`eos_token_id = [<|im_end|>,
<|endoftext|>]`) in every merged checkpoint. If a base model ships a valid config
(e.g. the 27B), it is preserved as-is. Details in
[ISSUES_AND_FIXES.md](ISSUES_AND_FIXES.md).

## Project notes

[ISSUES_AND_FIXES.md](ISSUES_AND_FIXES.md) — technical issues hit during SFT training
and clembench evaluation, and how they were resolved.
[DPO_REPORT.md](DPO_REPORT.md) — the DPO phase in detail: pair-data construction, the
pair-design iterations and their results, and the on-policy redesign.
