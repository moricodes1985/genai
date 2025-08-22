from dotenv import load_dotenv; load_dotenv()
import os

# NEW import
from langchain_openai import ChatOpenAI          # NEW import
from langchain.prompts import ChatPromptTemplate

api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# NEW arg names: model=..., api_key=...
chat = ChatOpenAI(model=model, temperature=0, api_key=api_key)

system = "You are an assistant that helps users find information about movies."
human = "Find information about the movie {movie_title}."
chat_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

# Runnable API
resp = (chat_prompt | chat).invoke({"movie_title": "Inception"})
print(resp.content)
