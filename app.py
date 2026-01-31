from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Setup
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "medibot"

app = Flask(__name__)

# Embeddings + Vector Store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# LLM (Ollama)
llm = Ollama(
    model="mistral",
    temperature=0
)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a medical assistant. Use the retrieved context to answer. "
        "If you don't know, say you don't know.\n\n{context}"
    ),
    ("human", "{input}")
])

# Helper
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG Chain
rag_chain = (
    {
        "context": lambda x: format_docs(
            retriever.invoke(x["input"])
        ),
        "input": lambda x: x["input"],
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_query = request.json.get("query")
    print("USER QUERY:", user_query)   # 👈 ADD THIS
    answer = rag_chain.invoke({"input": user_query})
    print("ANSWER:", answer)           # 👈 ADD THIS
    return jsonify({"answer": answer})

# Run
if __name__ == "__main__":
    app.run(host ="0.0.0.0", port = 8080, debug=True)


