# Exploration of the Potential of DPO in Dialogue Games

This repo presents an SFT + DPO post-training pipeline of Qwen3.5 models on [clembench](https://github.com/clp-research/clembench) / [Playpen](https://github.com/lm-playpen/playpen) dialogue-game benchmark. The best performing model which came out of this implementation was submitted to the [LM Playschool Challenge](https://lm-playschool.github.io).
The main research interest behind this project revolves around whether DPO can induce novel strategic and rule-following skills in dialogue-games playing LLMs, or whether it is better suited to refining behaviors already established via SFT. In order to conduct this investigation, multiple training ablations were constructed, targeting three skills which can easily impact performance on dialogue games, namely rule-following, strategic game-playing and excessive verbosity and rambling, and various combinations of the three.
The best performing model resulted from the training condition targeting only excessive verbosity, which took the supervised-fine-tuned Qwen3.5-9B model from 58.82 to **70.21 clemscore**. 
The results obtained point towards the direction of DPO being able to reliably reinforce behaviour the model already learnt in previous training stages, while being less effective when it comes to teaching completely new behaviours and strategy.


**Final model:** [`Makaco/lmps-challenge-qwen3.5-9b-dpo`](https://huggingface.co/Makaco/lmps-challenge-qwen3.5-9b-dpo). Consult the model card for more in-depth training details.
**DPO pairs dataset used to fine-tune the final model:** [`Makaco/lmps-challenge-dpo-pairs`](https://huggingface.co/datasets/Makaco/lmps-challenge-dpo-pairs)

## What's in the repo

```
fine_tuning/              training pipeline
  sft.py                  LoRA SFT on successful playpen rounds (assistant-only loss)
  dpo.py                  LoRA DPO on ready-made preference pairs (--pairs_files)
  merge_adapters.py       merge any adapter into its base. Fixes Qwen3.5 generation_config
dpo_pairs_construction/   building the preference pairs
  collect_onpolicy_pairs.py       harvest rounds from scored rollouts (by outcome)
  generate_onpolicy_comments.py   LLM judge reflection comments on aborted/failed rounds
  distill_onpolicy_pairs.py       comments -> concrete (chosen, rejected) pairs.
                                  Only the bare corrected moves, not the whole reflection, were kept in
                                  the final study 
  prompts.py                      the reflection prompts
  llm_wrapper.py                  LLM judges clients
data/dpo_pairs/           the exact pair files used for every reported run
eval_results/             scored summaries (clemscore/statscore) for every reported run (because of size
                          limitations, the whole results set with game-playing interactions will be uploaded on HF)
model_registry.json       clem model configs for all models in the study
game_registry.json        points clem at ../clembench
```

## Setup

```bash
# clembench (the games) must sit next to this repo - game_registry.json expects ../clembench
git clone https://github.com/clp-research/clembench.git
git clone https://github.com/xMakaco/dpo-potential-in-dialogue-games.git
cd dpo-potentials-in-dialogue-games

python3.10 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# needed to run evals on the val set of clembench games plus static benchmark (statscore):
pip install git+https://github.com/lm-playpen/playpen.git
```

Everything below was run on a single H100 in bf16; total training budget for the
best model is ~4.3 GPU-hours (see model card).

## Reproducing the best model

No API keys needed - the anti-verbosity path is fully self-contained (SFT data comes from
the HF Hub, the DPO pairs are in this repo and also published at [Makaco/lmps-challenge-dpo-pairs](https://huggingface.co/datasets/Makaco/lmps-challenge-dpo-pairs)). Defaults in the scripts are the exact
settings of the released model.

```bash
cd fine_tuning

# 1. SFT on successful playpen rounds
python sft.py --model Qwen/Qwen3.5-9B --grad_checkpointing \
  --output_dir checkpoints/sft/Qwen3.5-9B-all-linear

# 2. merge (also bakes in a correct generation_config - see note below)
python merge_adapters.py --adapter checkpoints/sft/Qwen3.5-9B-all-linear

# 3. anti-verbosity (anti-bleed) DPO from the merged SFT model
python dpo.py --model checkpoints/sft/Qwen3.5-9B-all-linear-merged \
  --pairs_files ../data/dpo_pairs/anti-verbosity.Qwen3.5-9B-sft.json \
  --output_dir checkpoints/dpo/Qwen3.5-9B-antibleed-only

# 4. merge again -> final model
python merge_adapters.py --adapter checkpoints/dpo/Qwen3.5-9B-antibleed-only
```

Or skip all of it and pull the released checkpoint from HF.

### How the pairs were built

In the clembench/playpen benchmarking environment, games are scored as aborted when the player model fails to adhere to specific formatting rules defined by a programmatic game master. On the other hand,
rounds where the model is able to avoid formatting mistakes, but still is unable to win the game are scored as failed. Won rounds are scored as successful.
This project aimed at constructing multiple sets of DPO pairs targeting three relevant skills for successfully conducting dialogue game rounds: rule-following, strategic game-playing and general rambling/excessive verbosity tendencies.
Pairs tackling rule-following and strategic game-playing were constructed by prompting an LLM judge (gpt-5.2-chat via Azure) to identify the mistaken move, generate a
reflection on what went wrong, and produce a corrected move. The corrected move
alone becomes the chosen response. The anti-verbosity pairs come instead from successful rounds, where chosen is the player model's own clean move, while rejected is the same move with one of five synthetic
verbose justifications appended.

Earlier pair designs put the reflection text inside the chosen response. However, the
reflections bled into the model's output style, making every response even more
verbose than usual and breaking clembench's strict parsers, with large clemscore drops.
This is why all final designs keep chosen as the bare, parser-valid move.

Across the ablations, DPO targeting strategic thinking was ineffective or
harmful, rule-adherence pairs helped mainly the smaller models (which abort more
often because of format violations), and the anti-verbosity condition alone gave the largest gain, consistent
with preference optimization refining behaviours the SFT model already has
rather than teaching new ones.

## The ablations

The pair types are the ablation axis. Every reported condition is just a choice of
`--pairs_files`; the trainer is identical.

| pairs | what they contain | on-policy? |
|---|---|---|
| `anti-verbosity.*` | chosen = the SFT model's own winning move; rejected = the same move with an appended verbose justification | fully |
| `aborted-rounds-pairs.*` | chosen = LLM-corrected move fixing a rule violation; rejected = the violating move. In all correction pairs, `chosen` is the bare move in the game's required format - the judge's reflection comment never enters the pair, so no external prose can bleed into the model's outputs | rejected is, chosen isn't |
| `failed-pairs-hidden-states.*` / `failed-pairs-no-hidden-states.*` | chosen = LLM judge's strategically better move in a lost round; rejected = the actual move (hidden-state games use a hindsight-guarded prompt - pass both files for the full "failed" condition) | rejected is, chosen isn't |
| `chosen_only.*` | the anti-verbosity chosen moves alone, as SFT data (via `sft.py --data_file`) - isolates the imitation component of the DPO gain | fully |

Combinations = multiple `--pairs_files`. Example, aborted + failed:

```bash
python dpo.py --model checkpoints/sft/Qwen3.5-9B-all-linear-merged \
  --pairs_files ../data/dpo_pairs/aborted-rounds-pairs.Qwen3.5-9B-sft.json \
                ../data/dpo_pairs/failed-pairs-hidden-states.Qwen3.5-9B-sft.json \
                ../data/dpo_pairs/failed-pairs-no-hidden-states.Qwen3.5-9B-sft.json \
  --output_dir checkpoints/dpo/Qwen3.5-9B-aborted-failed
```

Regenerating the pairs from scratch (instead of using `data/dpo_pairs/`) requires
rollouts (`clem run` + `clem score`) and, for the correction conditions, an Azure
OpenAI or DeepSeek key for the GPT passes:

```bash
cd dpo_pairs_construction
python collect_onpolicy_pairs.py --rollouts <scored-rollout-dir> --condition success   # anti-verbosity
python collect_onpolicy_pairs.py --rollouts <scored-rollout-dir> --condition aborted
python generate_onpolicy_comments.py --condition aborted --input ../data/dpo_pairs/onpolicy_aborted.<model>.json --model_id <judge>
python distill_onpolicy_pairs.py --condition aborted --input ../data/dpo_pairs/onpolicy_commented.aborted.<model>.<judge>.json --model_id <judge>
```

Evaluation instances are excluded from every harvest (`--exclude_eval`).
The shipped files in data/dpo_pairs/ were renamed for clarity, the scripts emit onpolicy_* names

## Evaluation
### To evaluate on the full clembench benchmark
```bash
clem run -g "{'benchmark':['2.0']}" -m <model_name>   # model_name from model_registry.json
clem score
```
### To evaluate on Playpen's val set
```bash
playpen eval <model_name> --suite all
```

Only the released model resolves from the HF Hub directly; all other registry entries
point to `fine_tuning/checkpoints/...` and expect checkpoints produced by the pipeline
above. Score summaries for every reported run are in `eval_results/`
(clemscore = (%played / 100) × quality, on the clembench 2.0 instances part of Playpen's val set).

## Results so far

| model | clemscore | statscore |
|---|---|---|
| Qwen3.5-9B SFT | 58.82 | 61.09 |
| **Qwen3.5-9B SFT + anti-verbosity DPO** | **70.21** | 58.73 |
| Qwen3.5-9B SFT + aborted-corrections DPO | 60.96 | 56.82 |
| Qwen3.5-9B chosen-only SFT | 65.36 | 59.35 |
| Qwen3.5-2B SFT | 38.05 | 42.88 |
| Qwen3.5-2B SFT + anti-verbosity DPO | 44.69 | 37.38 |
| Qwen3.5-2B SFT + aborted-corrections DPO | 54.28 | 42.11 |

Full per-game breakdowns in `eval_results/`; the complete ablation grid (failed
corrections and all combinations, plus the 2B scale point) is being finalised for the
technical report.

## The Qwen3.5 generation_config pitfall

Small Qwen3.5 releases ship without a `generation_config.json` and declare only
`<|endoftext|>` as EOS, so after fine-tuning the model never stops at the chat-turn
boundary (`<|im_end|>`) and hallucinates extra conversation turns, causing clembench's programmatic game master to break.
`merge_adapters.py` reconstructs a correct config (`eos_token_id = [<|im_end|>,
<|endoftext|>]`) in every merged checkpoint. If a base model ships a valid config
(e.g. the 27B), it is preserved as-is.
