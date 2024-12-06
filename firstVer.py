import os
import google.generativeai as genai

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\Eric\\Desktop\\degree\\gen-lang-client-0711781660-f9fc6f4fd7d4.json"
api_key = os.getenv("GEMINI_APIKEY")
genai.configure(api_key=api_key)

from google.cloud import aiplatform

aiplatform.init(project="gen-lang-client-0711781660")

from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(model="gemini-1.5-flash")

games = [
    "Space Blasters: Galactic Warfare. A sci-fi first-person shooter with stunning visuals",
    "Farmville Fantasy: Magical Crops. A relaxing farming simulation with a twist of magic",
    "Zombie Invasion: Nightfall. A thrilling zombie survival shooter with a dark narrative",
    "Puzzle Quest: Mind Benders. A brain-teasing puzzle game with cooperative multiplayer mode",
]

from langchain.prompts import PromptTemplate

PROMPT_TEMPLATE = """Here is the description of a game: "{game}".

Embed the game into the given text: "{text_to_personalize}".

Prepend a personalized message including the user's name "{user}" 
    and their preference "{preference}".

Make it sound engaging and fun.
"""

PROMPT = PromptTemplate(
    input_variables=["game","text_to_personalize","user","preference"],
    template=PROMPT_TEMPLATE
)

import langchain_experimental.rl_chain as rl_chain
import time
scoring_criteria_template = (
    "Given {preference} rank how good or bad this selection is {game}"
)

chain = rl_chain.PickBest.from_llm(
    llm=llm,
    prompt=PROMPT,
    selection_scorer=rl_chain.AutoSelectionScorer(
        llm=llm, scoring_criteria_template_str=scoring_criteria_template
    ),
)

for _ in range(5):
    try:
        response = chain.run(
            game=rl_chain.ToSelectFrom(games),
            user=rl_chain.BasedOn("Tom"),
            preference=rl_chain.BasedOn(["Non-shooter", "loves complex games", "loves strategy"]),
            text_to_personalize="This is the weeks specialty game, we \
                believe you will love it!",
        )
        time.sleep(10)
    except Exception as e:
        print(e)
    print(response["response"])
    selection_metadata=response["selection_metadata"]
    print(
        f"selected index: {selection_metadata.selected.index}, score: {selection_metadata.selected.score}"
    )
    print()


# class CustomSelectionScorer(rl_chain.SelectionScorer):
#     def score_response(
#         self, inputs, llm_response: str, event: rl_chain.PickBestEvent
#     ) -> float:
#         print(event.based_on)
#         print(event.to_select_from)

#         # you can build a complex scoring function here
#         # it is preferable that the score ranges between 0 and 1 but it is not enforced

#         selected_meal = event.to_select_from["meal"][event.selected.index]
#         print(f"selected meal: {selected_meal}")

#         if "Tom" in event.based_on["user"]:
#             if "Vegetarian" in event.based_on["preference"]:
#                 if "Chicken" in selected_meal or "Beef" in selected_meal:
#                     return 0.0
#                 else:
#                     return 1.0
#             else:
#                 if "Chicken" in selected_meal or "Beef" in selected_meal:
#                     return 1.0
#                 else:
#                     return 0.0
#         else:
#             raise NotImplementedError("I don't know how to score this user")