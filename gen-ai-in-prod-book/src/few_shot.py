
from typing import List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
import os
from dotenv import load_dotenv; load_dotenv()
# ---------- 1) Build your document store (RAG source) ----------
api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
emb = OpenAIEmbeddings(openai_api_key=api_key)
llm = ChatOpenAI(model=model, temperature=0, api_key=api_key)


#knowledge base
docs_texts = [
    "LangChain’s retrievers convert queries into embeddings and do vector search.",
    "FAISS is a fast similarity search library used as a vector store.",
    "Few-shot prompting provides demonstrations to steer model behavior.",
    "RAG augments the LLM with external, retrieved context to improve factuality.",
]
doc_metadatas = [{"id": i} for i in range(len(docs_texts))]
doc_store = FAISS.from_texts(docs_texts, embedding=emb, metadatas=doc_metadatas)
retriever = doc_store.as_retriever(search_kwargs={"k": 3})

def format_docs(docs: List):
    return "\n\n".join(d.page_content for d in docs)

# ---------- 2) Build your example bank + selector (Few-shot source) ----------
examples = [
    {
        "question": "How does RAG improve factual accuracy?",
        "answer": "It retrieves relevant passages from an external corpus and includes them in the prompt so the model grounds its answer."
    },
    {
        "question": "When should I use FAISS vs. Chroma?",
        "answer": "Use FAISS for fast in-memory similarity search; choose Chroma for a featureful, persistent local DB."
    },
    {
        "question": "What is a retriever in LangChain?",
        "answer": "An interface that, given a string query, returns a list of relevant documents."
    },
    {
        "question": "Why add few-shot examples?",
        "answer": "They steer style/format and clarify task boundaries by showing concrete demonstrations."
    },
]

# Use a separate FAISS index for examples so you can tune k independently
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=examples,
    embeddings=emb,
    vectorstore_cls=FAISS,
    k=2,  # number of examples to attach per query
)

example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="Q: {question}\nA: {answer}\n"
)

# --- 3) Compose one prompt that holds instructions + examples + context ---
prompt = FewShotPromptTemplate(
    example_selector=example_selector,      # dynamically picks examples per query
    example_prompt=example_prompt,
    input_variables=["context", "user_question"],
    example_separator="\n",                 # optional: how examples are joined
    prefix=(
        "You are a precise assistant. Use the context to answer the question.\n"
        "Cite facts only from the context. If missing, say you don't know.\n\n"
        "=== Examples (similar questions & ideal answers) ==="
        # ⬆️ DO NOT put {examples} here. FewShotPromptTemplate will inject them automatically.
    ),
    # Put context here so it appears AFTER the auto-inserted examples.
    suffix=(
        "\n=== Retrieved context ===\n{context}\n\n"
        "Q: {user_question}\nA:"
    ),
)


# NOTE: FewShotPromptTemplate will auto-insert the dynamically selected examples
# via {examples} in the prefix above.

# ---------- 4) Wire the RAG+FewShot pipeline with LCEL ----------
parse = StrOutputParser()

# Parallel branch: gather context and pass the raw user question through
rag_inputs = RunnableParallel(
    context=(retriever | format_docs),
    user_question=RunnablePassthrough()
)

# Full chain: retrieve -> format prompt (with examples) -> LLM -> text
chain = rag_inputs | prompt | llm | parse

# ---------- 5) Use it ----------
question = "Explain how a retriever and FAISS are used in a RAG pipeline."
answer = chain.invoke(question)
print(answer)
