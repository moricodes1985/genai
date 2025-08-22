from dotenv import load_dotenv; load_dotenv()
import os

# NEW import
from langchain_openai import ChatOpenAI          # NEW import
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader


api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# NEW arg names: model=..., api_key=...
llm = ChatOpenAI(model=model, temperature=0, api_key=api_key)

# Load the summarization chain
summarize_chain = load_summarize_chain(llm)

# Load the document using PyPDFLoader
document_loader = PyPDFLoader(file_path="file/AI_Agents.pdf")
document = document_loader.load()

# Summarize the document
summary = summarize_chain(document)
print(summary['output_text'])
