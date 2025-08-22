from dotenv import load_dotenv; load_dotenv()
import os
# NEW import
from langchain_openai import ChatOpenAI          # NEW import
from langchain.chains.summarize import load_summarize_chain
from newspaper import Article, build, Config
from langchain.schema import (
    HumanMessage
)

api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# NEW arg names: model=..., api_key=...
llm = ChatOpenAI(model=model, temperature=0, api_key=api_key)

headers = {
 'User-Agent': '''Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'''
}

article_url = """https://www.artificialintelligence-news.com/2022/01/25/meta-claims-new-ai-supercomputer-will-set-records/"""

url = "https://www.artificialintelligence-news.com/2022/01/25/meta-claims-new-ai-supercomputer-will-set-records/"

# Optional: set a desktop User-Agent & timeout (helps avoid 403s)
config = Config()
config.browser_user_agent = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
config.request_timeout = 10

article = Article(url, language="en", config=config)
article.download()   # fetch HTML (no need to use requests first)
article.parse()      # extract title, text, authors, publish_date, etc.

article_title = article.title
article_text = article.text

# prepare template for prompt
template ="""You are a very good assistant that summarizes online articles.

Here's the article you want to summarize.

==================
Title: {article_title}

{article_text}
==================

Write a summary of the previous article.
"""

prompt = template.format(article_title=article.title, article_text=article.text)

messages = [HumanMessage(content=prompt)]
summary = llm(messages)
print(summary.content)