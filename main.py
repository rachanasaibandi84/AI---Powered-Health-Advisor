from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
import requests
import os

# Setup your API keys
groq_api_key = os.getenv("GROQ_API_KEY")
rapidapi_key = os.getenv("RAPIDAPI_KEY")

# Define your LLM
llm = ChatGroq(temperature=0.7, groq_api_key=groq_api_key, model_name="llama3-8b-8192")

# Agent 1: Data Fetcher
fetcher = Agent(
    role="Data Fetcher",
    goal="Fetch current blood glucose data",
    backstory="Skilled in health APIs and data extraction",
    verbose=True,
    llm=llm
)

# Agent 2: Health Analyst
analyst = Agent(
    role="Health Analyst",
    goal="Give health recommendations based on glucose levels",
    backstory="Experienced doctor analyzing glucose data",
    verbose=True,
    llm=llm
)

# Define Tasks
task1 = Task(
    description="Fetch blood glucose data using RapidAPI",
    agent=fetcher
)

task2 = Task(
    description="Analyze the blood glucose data and provide recommendations",
    agent=analyst
)

# Assemble the Crew
crew = Crew(
    agents=[fetcher, analyst],
    tasks=[task1, task2],
    verbose=2
)

result = crew.kickoff()
print("Result:", result)
