import os
import google.generativeai as genai

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\Eric\\Desktop\\degree\\gen-lang-client-0711781660-f9fc6f4fd7d4.json"
api_key = os.getenv("GEMINI_APIKEY")
genai.configure(api_key=api_key)

from dotenv import load_dotenv
load_dotenv("C:\\Users\\Eric\\Desktop\\degree\\openweathermap.env")
api_key=os.environ.get("OPENWEATHERMAP_API_KEY")
os.environ["OPENWEATHERMAP_API_KEY"] = "6cabdf8f305f0b08c9e007f3ca5944e7"

from google.cloud import aiplatform

aiplatform.init(project="gen-lang-client-0711781660")

from langchain_google_vertexai import ChatVertexAI

model = ChatVertexAI(model="gemini-1.5-flash")

from langchain_community.utilities import OpenWeatherMapAPIWrapper

weather = OpenWeatherMapAPIWrapper()

AgentState = {}
AgentState["messages"]=[]

def function_1(state):
    messages = state['messages']
    user_input=messages[-1]
    complete_query = "Your task is to provide only the city name based on the user query. \
        Nothing more, just the city name mentioned. Following is the user query: " + user_input
    response = model.invoke(complete_query)
    state['messages'].append(response.content) #appends the AIMessage reply to the states
    return response.content

def function_2(state):
    messages = state['messages']
    agent_response = messages[-1]
    weather = OpenWeatherMapAPIWrapper()
    weather_data=weather.run(agent_response)
    state['messages'].append(weather_data)
    return weather_data

def function_3(state):
    messages = state['messages']
    user_input = messages[0]
    available_info = messages[-1]
    agent2_query = "Your task is to provide info concisely based on the user query and the available information from the internet. \
                        Following is the user query: " + user_input + " Available information: " + available_info
    response = model.invoke(agent2_query)
    return response.content 

from langgraph.graph import Graph


workflow = Graph()

workflow.add_node("agent", function_1)
workflow.add_node("tool", function_2)
workflow.add_node("responder", function_3)

workflow.add_edge('agent', 'tool')
workflow.add_edge('tool', 'responder')

workflow.set_entry_point("agent")
workflow.set_finish_point("responder")

app = workflow.compile()

