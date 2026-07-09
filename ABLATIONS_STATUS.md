# Ablation status tracker

A condition is ticked only when the full chain ran: **DPO training → merge → registry entry → clem eval**.
All runs start from the merged SFT model with clean pairs (`data/dpo_pairs/`), defaults of `dpo.py`
(β=0.3, 1 epoch, batch 2×4). Old tainted-data runs (aborted+failed 49.06, all 26.3, antibleed+verifstrat 57.0)
are superseded and not listed.

## Qwen3.5-9B

| # | condition | pairs files | done | clemscore | statscore |
|---|-----------|-------------|:----:|-----------|-----------|
| 0 | SFT baseline | — | ✅ | 58.82 | 61.09 |
| 1 | anti-verbosity | `anti-verbosity.Qwen3.5-9B-sft.json` | ✅ | **70.21** | 58.73 |
| 2 | aborted | `aborted-rounds-pairs.Qwen3.5-9B-sft.json` | ✅ | 60.96 | 56.82 |
| 3 | failed (hidden+verifiable) | `failed-pairs-hidden-states` + `failed-pairs-no-hidden-states` | ✅ | 44.43 | 61.55 |
| 4 | chosen-only SFT | `chosen_only.Qwen3.5-9B-sft.json` (via `sft.py --data_file`) | ✅ | 65.36 | 59.35 |
| 5 | aborted + failed | files of #2 + #3 | ✅ | 43.45 | 61.58 |
| 6 | anti-verbosity + aborted | files of #1 + #2 | ✅ | 63.69 | 60.28 |
| 7 | anti-verbosity + failed | files of #1 + #3 | ✅ | 42.71 | 58.54 |
| 8 | all three | files of #1 + #2 + #3 | ⬜ | | |

## Qwen3.5-2B

| # | condition | pairs files | done | clemscore | statscore |
|---|-----------|-------------|:----:|-----------|-----------|
| 0 | SFT baseline | — | ✅ | 38.05 | 42.88 |
| 1 | anti-verbosity | `anti-verbosity.Qwen3.5-2B-sft.json` | ✅ | 44.69 | 37.38 |
| 2 | aborted | `aborted-rounds-pairs.Qwen3.5-2B-sft.json` | ✅ | 54.28 | 42.11 |
| 2b | aborted (seed 123 replicate) | same as #2, `--seed 123` | ✅ | 51.03 | 44.96 |
| 3 | chosen-only SFT | `chosen_only.Qwen3.5-2B-sft.json` — **checkpoint merged, eval missing** | ⬜ | | |
| — | failed / combinations | blocked: 2B failed pairs are corrupt (Player-2 leak); clean first if wanted | ⬜ | | |

## Commands for the missing runs (run from `fine_tuning/`, H100 = `CUDA_VISIBLE_DEVICES=0`, check GPU is free first)

```bash
P=../data/dpo_pairs
AV=$P/anti-verbosity.Qwen3.5-9B-sft.json
AB=$P/aborted-rounds-pairs.Qwen3.5-9B-sft.json
FH=$P/failed-pairs-hidden-states.Qwen3.5-9B-sft.json
FV=$P/failed-pairs-no-hidden-states.Qwen3.5-9B-sft.json
M=checkpoints/sft/Qwen3.5-9B-all-linear-merged

# 5. aborted + failed
python dpo.py --model $M --pairs_files $AB $FH $FV --output_dir checkpoints/dpo/Qwen3.5-9B-aborted-failed-clean

# 6. anti-verbosity + aborted
python dpo.py --model $M --pairs_files $AV $AB --output_dir checkpoints/dpo/Qwen3.5-9B-antiverbosity-aborted

# 7. anti-verbosity + failed
python dpo.py --model $M --pairs_files $AV $FH $FV --output_dir checkpoints/dpo/Qwen3.5-9B-antiverbosity-failed

# 8. all three
python dpo.py --model $M --pairs_files $AV $AB $FH $FV --output_dir checkpoints/dpo/Qwen3.5-9B-all-clean

# after each training: merge, then eval (registry entry needed before playpen eval)
python merge_adapters.py --adapter checkpoints/dpo/<name>
playpen eval --suite all <registry-model-name>     # from repo root; writes playpen-eval/<timestamp>/

# 2B #3: eval only (checkpoint already merged and registered)
playpen eval --suite all Qwen3.5-2B-chosen-only-merged
```

After each eval: tick the row here, fill the scores, copy `*.val.json` + `clem/results.{csv,html}` +
`static/results.{csv,html}` into `eval_results/<run-name>/`, and add the registry entry for the new
merged checkpoint (ask Claude — same template as `Qwen3.5-9B-dpo-failed-hidden-verifiable-merged`).
