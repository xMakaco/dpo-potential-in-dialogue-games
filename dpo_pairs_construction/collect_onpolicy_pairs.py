"""Script to collect on-policy DPO material from scored clembench v2.0 rollouts of the merged SFT model.
Requires running clem run -g "{'benchmark':['2.0']}" -m <model>, then clem score to obtain the scored rollouts and the outcomes of the rounds.
The script extracts rounds based on outcome and generates a json file with the records for the
DPO training ablations: aborted gives one record per rule violation (for the corrections step),
failed gives one record per player per lost round (the suboptimal move is located later by an
LLM judge), success gives ready-to-train anti-verbosity pairs.
Instances present in the playpen-eval set passed via --exclude_eval are skipped.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

_HERE = Path(__file__).parent
OUT_DIR = _HERE / "dpo_pairs"

# Event types that indicate a rule violation (for the aborted condition),
# as per clembench's prorammatic game master.
VIOLATION_TYPES = {"invalid format", "invalid response", "validation error"}

# Syntetic extra prose to be added to successful rounds in order to
# generate anti-verbosity/rambling DPO pairs.
JUSTIFICATIONS = [
    "\nThis response follows the required format and avoids any rule violations.",
    "\nThis is a strategically sound choice given the current game state.",
    " (this answer is correct because it matches the information given earlier)",
    "\nNote: the required format was respected and no forbidden words were used.",
    "\nThe reasoning behind this choice is that it narrows down the options efficiently.",
]


def pair_id(*parts):
    return hashlib.sha1("|".join(str(p)
                        for p in parts).encode()).hexdigest()[:16]


def eval_instance_keys(eval_dirs):
    keys = set()
    for d in eval_dirs:
        for inst in Path(d).rglob("instance_*"):
            if inst.is_dir():
                keys.add((inst.parts[-3], inst.parts[-2], inst.name))
    return keys


def read_outcome(scores_path: Path):
    """Return the outcome of a scored clembench instance as either 'aborted, 'failed', success'
    or None if unscored.
    Note: clembench's key for failed rounds is Lose, but it is mapped here to playpen's 'failed' terminology for consistency.
    """
    if not scores_path.exists():
        return None
    epi = json.load(open(scores_path)).get("episode scores", {})
    if epi.get("Aborted") in (1, True):
        return "aborted"
    if epi.get("Success") in (1, True):
        return "success"
    if epi.get("Lose") in (1, True):
        return "failed"
    ms = epi.get("Main Score")
    if isinstance(ms, (int, float)) and ms == ms:
        return "success" if ms >= 100 else "failed"
    return None


def real_error_message(action):
    """Extract the human-readable rejection reason from a violation event."""
    content = action.get("content", "")
    # codenames: content is a dict with the reason under 'type'
    if isinstance(content, dict):
        return str(content.get("type") or content.get(
            "error") or content), content.get("player")
    return str(content), None


def model_players_of(scored_dict):
    """Extract the role that was actually played by the model in the game.
    Exclude the GM and other programmatic roles (e.g. textmapworld's PathDescriber) that emit scripted state text, not model decisions,
    so they must not become training records.
    """
    players = scored_dict.get("players", {})
    return {p for p, info in players.items()
            if p != "GM" and info.get("model_name") != "programmatic"}


def per_player_histories(events, model_players):
    """Reconstruct each model player's chat history (skips programmatic roles)."""
    histories = {}
    for msg in events:
        action = msg.get("action", {})
        a_type = action.get("type")
        content = action.get("content", "")
        src, dst = msg.get("from"), msg.get("to")
        if a_type == "send message" and src == "GM" and dst in model_players:
            histories.setdefault(dst, []).append(
                {"role": "user", "content": content})
        elif a_type == "get message" and src in model_players:
            histories.setdefault(src, []).append(
                {"role": "assistant", "content": content})
    return histories


def collect_aborted(events, game, experiment, instance, model_players):
    """One violation record per rejection in a aborted rounds."""
    histories = {}
    last_response = None
    last_error = None
    records = []

    for msg in events:
        action = msg.get("action", {})
        a_type = action.get("type")
        content = action.get("content", "")
        src, dst = msg.get("from"), msg.get("to")

        if a_type == "send message" and src == "GM" and dst in model_players:
            histories.setdefault(dst, []).append(
                {"role": "user", "content": content})
        elif a_type == "get message" and src in model_players:
            ctx = list(histories.get(src, []))
            last_response = (src, content, ctx)
            histories.setdefault(src, []).append(
                {"role": "assistant", "content": content})
        elif a_type == "metadata" and str(content).startswith("Error"):
            last_error = (str(content), None)
        elif a_type in VIOLATION_TYPES and last_response is not None:
            reason, player_hint = real_error_message(action)
            # prefer a descriptive 'Error:' metadata if the marker has none
            if (not reason or reason.lower().startswith(
                    "game_result")) and last_error:
                reason = last_error[0]
            player, response, ctx = last_response
            records.append({
                "pair_id": pair_id(game, experiment, instance, player, len(ctx)),
                "meta": {"outcome": "aborted", "game": game, "experiment": experiment,
                         "instance": instance, "player": player},
                "messages": ctx + [{"role": "assistant", "content": response}],
                # aux fields for the distill step
                "rejected": response,
                "gm_error": reason,
            })
            last_response = None
    return records


def collect_failed(events, game, experiment, instance, model_players):
    """One per-player record for a failed round, for the LLM judge to localize."""
    histories = per_player_histories(events, model_players)
    records = []
    for player, hist in histories.items():
        moves = [m["content"] for m in hist if m["role"] == "assistant"]
        if not moves:
            continue
        records.append({
            "pair_id": pair_id(game, experiment, instance, player, "failed"),
            "meta": {"outcome": "failed", "game": game, "experiment": experiment,
                     "instance": instance, "player": player},
            "messages": hist,
            "moves": moves,
        })
    return records


def collect_antibleed(
        events,
        game,
        experiment,
        instance,
        rng,
        per_instance,
        model_players):
    """Collect successful rounds and generated rejected entries in the DPO pairs by appending one of the 5 justifications."""
    histories = per_player_histories(events, model_players)
    out = []
    for player, hist in histories.items():
        cands = []
        ctx = []
        for m in hist:
            if m["role"] == "assistant":
                cands.append((list(ctx), m["content"]))
            ctx.append(m)
        rng.shuffle(cands)
        for context, move in cands[:per_instance]:
            corrupted = move.rstrip() + rng.choice(JUSTIFICATIONS)
            out.append({
                "pair_id": pair_id(game, experiment, instance, player, len(context), "ab"),
                "prompt": context,
                "chosen": [{"role": "assistant", "content": move}],
                "rejected": [{"role": "assistant", "content": corrupted}],
            })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollouts",
        required=True,
        help="scored clem results dir, e.g. onpolicy-rollouts/<model>")
    parser.add_argument(
        "--condition",
        required=True,
        choices=[
            "aborted",
            "failed",
            "success"],
        help="Which scored outcome to harvest (failed = clembench 'Lose').")
    parser.add_argument(
        "--exclude_eval",
        nargs="*",
        default=[],
        help="playpen-eval run dir(s) whose instances must be excluded.")
    parser.add_argument("--antibleed_per_instance", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rollouts = Path(args.rollouts)
    model_name = rollouts.name
    excluded = eval_instance_keys(args.exclude_eval)
    print(f"Excluding {len(excluded)} instances seen by local playpen evals.")

    rng = random.Random(args.seed)
    out_records = []
    n_inst = n_match = n_unscored = n_skipped = 0
    games = {}

    for inter in sorted(rollouts.rglob("interactions.json")):
        game, experiment, instance = inter.parts[-4], inter.parts[-3], inter.parts[-2]
        if (game, experiment, instance) in excluded:
            n_skipped += 1
            continue
        n_inst += 1
        outcome = read_outcome(inter.parent / "scores.json")
        if outcome is None:
            n_unscored += 1
            continue
        if outcome != args.condition:
            continue
        n_match += 1
        scored = json.load(open(inter))
        model_players = model_players_of(scored)
        events = [m for turn in scored.get("turns", []) for m in turn]
        if args.condition == "aborted":
            recs = collect_aborted(
                events, game, experiment, instance, model_players)
        elif args.condition == "failed":
            recs = collect_failed(
                events, game, experiment, instance, model_players)
        else:
            recs = collect_antibleed(
                events,
                game,
                experiment,
                instance,
                rng,
                args.antibleed_per_instance,
                model_players)
        out_records.extend(recs)
        if recs:
            games[game] = games.get(game, 0) + len(recs)

    OUT_DIR.mkdir(exist_ok=True)
    fname = {
        "aborted": f"onpolicy_aborted.{model_name}.json",
        "failed": f"onpolicy_failed_rounds.{model_name}.json",
        "success": f"onpolicy_antibleed.{model_name}.json",
    }[args.condition]
    out_path = OUT_DIR / fname
    with open(out_path, "w") as f:
        json.dump(out_records, f, indent=2)

    print(f"\nCondition: {args.condition}")
    print(
        f"Instances scanned: {n_inst} (unscored: {n_unscored}, excluded: {n_skipped})")
    print(f"Instances matching outcome: {n_match}")
    print(f"Records written: {len(out_records)} -> {out_path}")
    for g, n in sorted(games.items(), key=lambda x: -x[1]):
        print(f"  {g}: {n}")


if __name__ == "__main__":
    main()
