# Imports
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


# Load env
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found")


# -------- FIX 1: define load_pdf_file ----------
def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()


# -------- FIX 2: create text_splitter ----------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20
)


# -------- FIX 3: embeddings (no function call) ----------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -------- YOUR ORIGINAL LOGIC (fixed) ----------
extracted_data = load_pdf_file(data="Data/")
text_chunks = text_splitter.split_documents(extracted_data)

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medibot"

existing_indexes = [i["name"] for i in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

