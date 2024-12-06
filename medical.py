import os
import google.generativeai as genai

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\Eric\\Desktop\\degree\\gen-lang-client-0711781660-f9fc6f4fd7d4.json"
api_key = os.getenv("GEMINI_APIKEY")
genai.configure(api_key=api_key)

from google.cloud import aiplatform

aiplatform.init(project="gen-lang-client-0711781660")

from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(model="gemini-1.5-flash")

medicines = [
    "Cetirizine: An antihistamine used for relieving allergy symptoms like runny nose and itchy eyes, suitable for mild throat discomfort.",
    "Paracetamol: A common pain reliever used to reduce fever and alleviate mild to moderate pain, gentle on sensitive individuals.",

    "Ibuprofen: An anti-inflammatory drug that helps with pain, fever, and inflammation but might be too harsh for someone sensitive.",
    "Amoxicillin: An antibiotic used to treat bacterial infections such as respiratory infections and ear infections, unnecessary for viral throat issues.",
    "Aspirin: A nonsteroidal anti-inflammatory drug (NSAID) that can irritate the stomach and may not be ideal for mild symptoms.",
    "Codeine: A strong pain reliever that can cause drowsiness and may be too strong for mild throat discomfort.",
    "Prednisone: A corticosteroid that can have significant side effects, unnecessary for mild throat pain.",
    "Diphenhydramine: A sedating antihistamine that might cause drowsiness, making it unsuitable for daytime use.",
    "Ciprofloxacin: A broad-spectrum antibiotic that isn't suitable unless there's a confirmed bacterial infection."
]

from langchain.prompts import PromptTemplate

PROMPT_TEMPLATE = """Here is the description of a medicine: "{medicine}".

Embed the medicine into the following text: "{text_to_personalize}".

Prepend a personalized message for the user, including their name "{user}", their medical preference "{preference}", and their symptoms "{symptoms}".

You are a doctor with 20 years of experience, skilled in clearly explaining the diagnosis process. Ensure the tone is professional, yet easy to understand for someone without prior medical knowledge. Be empathetic and patient-focused, offering reassurance throughout. 
Use the symptoms provided to guide your explanation and offer relevant advice.
"""

PROMPT = PromptTemplate(
    input_variables=["medicine","text_to_personalize","user","preference","symptoms"],
    template=PROMPT_TEMPLATE
)

import langchain_experimental.rl_chain as rl_chain
import time
# prompt to give the llm to score ur actual prompt/response
scoring_criteria_template = (
    "Given the user's preference of {preference} and symptoms of {symptoms}, please rate how appropriate and effective this medicine selection is: {medicine}. Consider factors such as how well it aligns with the user's needs, the suitability of the medicine for their condition, and any potential side effects or benefits."
)

class CustomSelectionScorer(rl_chain.SelectionScorer):
    def score_preference(self, preference, selected_medicine):
        score = 0.0
        if "natural remedies" in preference:
            if "natural" in selected_medicine.lower():
                score += 0.6
        if "sensitive to strong medications" in preference:
            if any(word in selected_medicine.lower() for word in ["gentle", "mild"]):
                score += 0.4
        return min(score, 1.0)

    def score_symptoms(self, symptoms, selected_medicine):
        if "throat discomfort" in symptoms:
            if "throat" in selected_medicine.lower() or "allergy" in selected_medicine.lower():
                return 0.8
            if "broad-spectrum" in selected_medicine.lower():
                return 0.2
        return 0.0

    def score_response(
        self, inputs, llm_response: str, event: rl_chain.PickBestEvent
    ) -> float:
        selected_medicine = event.to_select_from["medicine"][event.selected.index]
        preference_score = self.score_preference(event.based_on["preference"], selected_medicine)
        symptom_score = self.score_symptoms(event.based_on["symptoms"], selected_medicine)

        # Combine the scores with weighted importance
        final_score = (0.7 * symptom_score) + (0.3 * preference_score)
        return final_score

# custom scoring but automated, can save time/money since llm is prompted less
chain = rl_chain.PickBest.from_llm(
    llm=llm,
    prompt=PROMPT,
    selection_scorer=CustomSelectionScorer(),
)

# this is using the score from the llm, probably inaccurate
# chain = rl_chain.PickBest.from_llm(
#     llm=llm,
#     prompt=PROMPT,
#     selection_scorer=rl_chain.AutoSelectionScorer(
#         llm=llm, scoring_criteria_template_str=scoring_criteria_template
#     ),
# )

# this is for custom scoring, for human to score so more accurate but time-consuming and not automated
# chain = rl_chain.PickBest.from_llm(
#     llm=llm,
#     prompt=PROMPT,
#     selection_scorer=rl_chain.AutoSelectionScorer(
#         llm=llm, scoring_criteria_template_str=None
#     ),
# )

for _ in range(5):
    try:
        response = chain.run(
            medicine=rl_chain.ToSelectFrom(medicines),
            user=rl_chain.BasedOn("Jon Jones"),
            preference=rl_chain.BasedOn(["prefers natural remedies", "has a mild sensitivity to strong medications", "seeks relief for throat discomfort"]),
            symptoms=rl_chain.BasedOn(["I think I had some uncomfortable feeling in my throat but I'm not sure how to describe it"]),
            text_to_personalize="Based on your symptoms and preferences, we've identified a treatment option that we believe will suit you. Let us know what you think!",
        ) 
        time.sleep(10)
    except Exception as e:
        print(e)

    print(response["response"])
    selection_metadata=response["selection_metadata"]


    # remember to do 0.0 or 1.0, must have decimal
    # user_score = float(input("Rate the suitability of this suggestion (0-1): "))
    # chain.update_with_delayed_score(score=user_score, chain_response=response, force_score=True)



    print(
        f"selected index: {selection_metadata.selected.index}, score: {selection_metadata.selected.score}"
    )
    print()

chain.save_progress()
# Don't really know how it actually scores, medicines that should not fit that well are getting recommended and getting high score
# One cool thing is that it recognize that the person prefers natural remedies and suggested a natural remedy
# scoring manually https://github.com/langchain-ai/langchain-experimental/blob/7177f464139a0e91227e233eab7b754b221d46aa/libs/experimental/langchain_experimental/rl_chain/base.py#L454
# line 339?
#