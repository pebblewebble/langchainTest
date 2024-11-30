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

chain = rl_chain.PickBest.from_llm(llm=llm,prompt=PROMPT)

for _ in range(5):
    try:
        response = chain.run(
            game=rl_chain.ToSelectFrom(games),
            user=rl_chain.BasedOn("Tom"),
            preference=rl_chain.BasedOn(["Non-shooter", "loves complex games", "loves strategy"]),
            text_to_personalize="This is the weeks specialty game, we \
                believe you will love it!",
        )
    except Exception as e:
        print(e)
    print(response["response"])
    print()