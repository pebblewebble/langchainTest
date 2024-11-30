import os
import google.generativeai as genai

api_key = os.getenv("GEMINI_APIKEY")
genai.configure(api_key=api_key)

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

response = chain.run(
    game=rl_chain.ToSelectFrom(games),
    user=rl_chain.BasedOn("Tom"),
    preference=rl_chain.BasedOn(["Non-shooter", "loves complex games"]),
    text_to_personalize="This is the weeks specialty game, we \
        believe you will love it!",
)

print(response["response"])