import json
import os
from llm_wrapper import QwenLocalClient
# Turn into a class
def build_prompt(entry, outcome):
    if outcome == "aborted":

        system_prompt = "You are a highly capable AI assistant. " \
        "You will be shown multiple examples of cases where another model failed at playing a variety of dialogue-based games, because it was unable to adhere to their basic rules. " \
        "The games in the roster are: taboo, hot_air_balloon, referencegame, adventuregame, wordle, wordle_withclue, wordle_withcritic, privateshared, dond, guesswhat, " \
        "textmapworld_graphreasoning, textmapworld, matchit_ascii, wordle_withclue, textmapworld_specificroom, imagegame, codenames. " \
        "For each game, you will be provided with its rules, along with instances of gameplays from another model. " \
        "All the games stopped early because the model playing them made rules violations which ended in the round being aborted. " \
        "Your role is to add, at the end of each game round, a verbal feedback regarding what the other model did wrong, which rules were violated and why, and what could have been done better. " 
        "Be concise and systematic in providing your reflection. Do not address the model or the players directly. For instance, don't start with The model violated rule X, but rather during the round rule X was violated, or simply Rule X was violated."
        "Produce short and focused sentences for which rules were violated, how they were violated and an example of a better action to be taken in the game."
        "Since some of the games are collaborative, you might sometimes have to evaluated what violated the rules inside of dialogue exchanges between two players. " \
        "Be expressive and clear in your comment, as it should be used to teach other players how to stick to the rules. " \
        "Always start with [COMMENT]:"

        if entry["meta"]["outcome"] != "aborted":
            return None
        
        rules = entry["messages"][0]["content"]

        entry_cleaned= {k: v for k, v in entry.items() if k != "meta"}
        game_round = json.dumps(entry_cleaned, indent=2)
        messages_content = f"Here are the GAME RULES: {rules}\nHere is an ABORTED GAME ROUND: {game_round}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": messages_content}
        ]

    elif outcome == "failure":
        # Change system prompt 
        system_prompt = "You are a highly capable AI assistant. " \
        "You will be shown multiple examples of cases where another model failed at playing a variety of dialogue-based games, because it was unable to make strategic choices which could lead to winning the game round. " \
        "The games in the roster are: taboo, hot_air_balloon, referencegame, adventuregame, wordle, wordle_withclue, wordle_withcritic, privateshared, dond, guesswhat, " \
        "textmapworld_graphreasoning, textmapworld, matchit_ascii, wordle_withclue, textmapworld_specificroom, imagegame, codenames. " \
        "For each game, you will be provided with its rules, along with instances of gameplays from another model. " \
        "All the games ended in failure because the model or models playing it were not able to make strategically sound choices and therefore did not reach victory in the predetermined number of attempts they were allowed." \
        "Your role is to identify, in the trascript of vaiorus game rounds, the first moment were a suboptimal choice which ultimately lead to failure was made."
        "After you have found it, add, after the message exchange were the suboptimal choice was made, a verbal feedback regarding what the other model did wrong, which choice was strategically suboptimal and why, and what could have been done better. " \
        "Do so not only for the first instance of a strategically suboptimal choice, but also for all the others that followed (according to yoiur judgment, there might be only one suboptimal choice or many)."
        "Be concise and systematic in providing your reflection, which should be informative but also straight to the point. Do not address the model or the players directly. For instance, don't start with The model chose action X which is suboptimal because, but rather During the round suboptimal action X was taken, or simply Choice X was suboptimal because."
        "Produce short and focused sentences for which suboptimal choices were made, why they were suboptimal and an example of a better action to be taken in the game. Always produce at least one example of a strategically better choice which could proficiently replace the suboptimal one."
        "Since some of the games are collaborative, you might sometimes have to evaluated what violated the rules inside of dialogue exchanges between two players. " \
        "Be expressive and clear in your comment, as it should be used to teach other players how to stick to the rules. " \
        "Do not address the players, simply provide a comment or reasoning as instructed. Always start with [COMMENT]:"

        if entry["meta"]["outcome"] != "aborted":
            return None
        
        rules = entry["messages"][0]["content"]

        entry_cleaned= {k: v for k, v in entry.items() if k != "meta"}
        game_round = json.dumps(entry_cleaned, indent=2)
        messages_content = f"Here are the GAME RULES: {rules}\nHere is an ABORTED GAME ROUND: {game_round}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": messages_content}
        ]


    return messages

def main():
    model = QwenLocalClient(model_id="Qwen/Qwen3.6-35B-A3B")

    with open('../failures_data.json') as f:
        dataset = json.load(f)

    processed_dataset = []

    for entry in dataset:
        messages = build_prompt(entry, outcome="aborted")
        if messages is None:
            continue

        try:
            print("Generating corrections for game rounds")
            assistant_comment = model.generate_comment(messages)
            
            entry["messages"].append({
                "role": "assistant",
                "content": assistant_comment
            })
            processed_dataset.append(entry)

        except Exception as e:
            print(f"Skipping entry due to error: {e}")
            continue

    with open('../failures_with_comments.json', 'w') as f:
        json.dump(processed_dataset, f, indent=2)
    print("Processing complete!")

if __name__ == "__main__":
    main()