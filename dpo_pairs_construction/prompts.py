"""Defines the prompts that will be used to generate reflection comments on on-policy aborted or failed rounds
using an LLM judge, used by generate_onpolicy_comments.py.
build_prompt selects the prompt based on the outcome of interest.
A specific prompt to handle failed rounds of hidden-state exploration games is also provided.
"""

import json


HIDDEN_STATE_GAMES = {
    "textmapworld",
    "textmapworld_specificroom",
    "textmapworld_graphreasoning",
    "adventuregame",
}

HIDDEN_STATE_FAILURE_PROMPT = "You are a highly capable AI assistant. " \
    "You will be shown multiple examples of cases where another model failed at playing a variety of dialogue-based games, because it was unable to make strategic choices which could lead to winning the game round. " \
    "The games in the roster are: " \
    "textmapworld_graphreasoning, textmapworld, textmapworld_specificroom, adventuregame. " \
    "For each game, you will be provided with its rules, along with instances of gameplays from another model. " \
    "All the games ended in failure because the model or models playing it were not able to make strategically sound choices and therefore did not reach victory in the predetermined number of attempts they were allowed. " \
    "Note that these games are exploration-based, therefore naturally contain hidden states: the player doesn't have access to all information at once, as it is not provided a map of the area that it is expected to explore. " \
    "Your role is to identify, in the trascript of various game rounds, the suboptimal choices which ultimately lead to failure was made. " \
    "Contrary to the player, you will have access to the whole state of the game at all times, since you will be provided the complete transcript. of the game round. " \
    "From the end of the transcript, it will be possible for you to reconstruct the whole map and therefore understand which choices were suboptimal in the context. " \
    "You can use this information to judge whether a specific move is legal given the rules of the game, however only flag a move as a mistake if a better choice was " \
    "identifiable with the information AVAILABLE AT THAT TURN; do not justify a corrected move using later-revealed knowledge. " \
    "For example, do not provide a correction such as The right move would have been GO: NORTH if only a complete knowledge of the map would justify this choice. " \
    "An example of a correction that you could make, on the other hand, is Avoid going back to the direction one came from, since that is clearly a poor strategic choice regardless of the knowledge about the map." \
    "After you have reviewed the round and identified the suboptimal choices, add, after the message exchange where the suboptimal choice was made, a verbal feedback regarding what the other model did wrong, which choice was strategically suboptimal and why, and what could have been done better. " \
    "Again, do so only if the information needed to take the choice could have been available to the player at that specific time in the game. " \
    "Be concise and systematic in providing your reflection, which should be informative but also straight to the point. " \
    "Write in impersonal third person throughout. Never use 'you', 'the model', or 'the assistant' as the subject of a sentence — the subject should always be the rule, the choice, the action, or the round. For example, write 'A suboptimal choice was made at turn 3' or 'The strategic value of X was overlooked', not 'The model made a suboptimal choice' or 'You overlooked X'. " \
    "Produce short and focused sentences for which suboptimal choices were made, why they were suboptimal and an example of a better action to be taken in the game. Always produce at least one example of a strategically better choice which could proficiently replace the suboptimal one. " \
    "Be expressive and clear in your comment, as it should be used to teach other players how to stick to the rules. " \
    "Always start with [COMMENT]: followed by your reflection. " \
    "Example format:\n[COMMENT]: <concise reflection on the suboptimal choice, why it was suboptimal, and a concrete better alternative.> "


def build_prompt(entry, outcome):
    if outcome == "aborted":

        system_prompt = "You are a highly capable AI assistant. " \
            "You will be shown multiple examples of cases where another model failed at playing a variety of dialogue-based games, because it was unable to adhere to their basic rules. " \
            "The games in the roster are: taboo, hot_air_balloon, referencegame, adventuregame, wordle, wordle_withclue, wordle_withcritic, privateshared, dond, guesswhat, " \
            "textmapworld_graphreasoning, textmapworld, matchit_ascii, wordle_withclue, textmapworld_specificroom, imagegame, codenames. " \
            "For each game, you will be provided with its rules, along with instances of gameplays from another model. " \
            "All the games stopped early because the model playing them made rules violations which ended in the round being aborted. " \
            "Your role is to add, at the end of each game round, a verbal feedback regarding what the other model did wrong, which rules were violated and why, and what could have been done better. " \
            "Be concise and systematic in providing your reflection. Do not address the model or the players directly. For instance, don't start with The model violated rule X, but rather during the round rule X was violated, or simply Rule X was violated." \
            "Produce short and focused sentences for which rules were violated, how they were violated and an example of a better action to be taken in the game." \
            "Since some of the games are collaborative, you might sometimes have to evaluated what violated the rules inside of dialogue exchanges between two players. " \
            "Be expressive and clear in your comment, as it should be used to teach other players how to stick to the rules. " \
            "Always start with [COMMENT]: followed by your reflection. " \
            "Example format:\n[COMMENT]: <concise reflection on the violated rule, how it was violated, and a concrete better alternative.> "

        if entry["meta"]["outcome"] != "aborted":
            return None

        rules = entry["messages"][0]["content"]

        entry_cleaned = {k: v for k, v in entry.items() if k != "meta"}
        game_round = json.dumps(entry_cleaned, indent=2)
        messages_content = f"Here are the GAME RULES: {rules}\nHere is an ABORTED GAME ROUND: {game_round}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": messages_content}
        ]

    elif outcome == "failed":
        system_prompt = "You are a highly capable AI assistant. " \
            "You will be shown multiple examples of cases where another model failed at playing a variety of dialogue-based games, because it was unable to make strategic choices which could lead to winning the game round. " \
            "The games in the roster are: taboo, hot_air_balloon, referencegame, adventuregame, wordle, wordle_withclue, wordle_withcritic, privateshared, dond, guesswhat, " \
            "textmapworld_graphreasoning, textmapworld, matchit_ascii, wordle_withclue, textmapworld_specificroom, imagegame, codenames. " \
            "For each game, you will be provided with its rules, along with instances of gameplays from another model. " \
            "All the games ended in failure because the model or models playing it were not able to make strategically sound choices and therefore did not reach victory in the predetermined number of attempts they were allowed." \
            "Your role is to identify, in the trascript of vaiorus game rounds, the first moment where a suboptimal choice which ultimately lead to failure was made." \
            "After you have found it, add, after the message exchange were the suboptimal choice was made, a verbal feedback regarding what the other model did wrong, which choice was strategically suboptimal and why, and what could have been done better. " \
            "Do so not only for the first instance of a strategically suboptimal choice, but also for all the others that followed (according to your judgment, there might be only one suboptimal choice or many)." \
            "Do noy write your reflections all together at the end, but rather below each suboptimal choice you detect." \
            "Be concise and systematic in providing your reflection, which should be informative but also straight to the point. " \
            "Write in impersonal third person throughout. Never use 'you', 'the model', or 'the assistant' as the subject of a sentence — the subject should always be the rule, the choice, the action, or the round. For example, write 'A suboptimal choice was made at turn 3' or 'The strategic value of X was overlooked', not 'The model made a suboptimal choice' or 'You overlooked X'. " \
            "Produce short and focused sentences for which suboptimal choices were made, why they were suboptimal and an example of a better action to be taken in the game. Always produce at least one example of a strategically better choice which could proficiently replace the suboptimal one." \
            "Since some of the games are collaborative, you might sometimes have to evaluated what violated the rules inside of dialogue exchanges between two players. " \
            "Be expressive and clear in your comment, as it should be used to teach other players how to stick to the rules. " \
            "Always start with [COMMENT]: followed by your reflection. " \
            "Example format:\n[COMMENT]: <concise reflection on the suboptimal choice, why it was suboptimal, and a concrete better alternative.> "

        if entry["meta"]["outcome"] != "failed":
            return None

        # Hidden-state exploration games: use the end-of-round, hindsight-guarded
        # prompt instead of the generic turn-by-turn one.
        if entry["meta"]["game"] in HIDDEN_STATE_GAMES:
            system_prompt = HIDDEN_STATE_FAILURE_PROMPT

        rules = entry["messages"][0]["content"]

        entry_cleaned = {k: v for k, v in entry.items() if k != "meta"}
        game_round = json.dumps(entry_cleaned, indent=2)
        messages_content = f"Here are the GAME RULES: {rules}\nHere is a FAILED GAME ROUND: {game_round}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": messages_content}
        ]

    return messages
