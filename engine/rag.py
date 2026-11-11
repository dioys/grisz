"""Reads lesson PDFs into Chroma (embeddings) and hands back a PDF's text, chunk by chunk, with page labels."""
import os
from functools import cache
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"        # lesson PDFs, one subject each
STORAGE = ROOT / "storage"  # embedding model (+ local Chroma files outside Docker)

load_dotenv(ROOT / ".env")
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir=str(STORAGE / "models"))

CALLBACKS = []  # passed to every LangGraph run; empty = no tracing
if os.getenv("LANGFUSE_PUBLIC_KEY"):  # tracing is on only when Langfuse keys are in .env
    from langfuse.langchain import CallbackHandler
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    LlamaIndexInstrumentor().instrument()  # LlamaIndex steps: reading and embedding PDFs
    CALLBACKS = [CallbackHandler()]        # LangGraph steps: exam writer, checker


@cache
def collection():
    host = os.getenv("CHROMA_HOST")  # set in Docker: the shared Chroma server; unset: local files
    client = chromadb.HttpClient(host=host, port=8000) if host else chromadb.PersistentClient(path=str(STORAGE / "chroma"))
    return client.get_or_create_collection("modules")


@cache
def load_index():
    """Read the PDFs once and save their embeddings in Chroma; later runs reuse them."""
    store = ChromaVectorStore(chroma_collection=collection())
    if collection().count():
        return VectorStoreIndex.from_vector_store(store)
    docs = SimpleDirectoryReader(DATA, required_exts=[".pdf"]).load_data()  # one document per PDF page
    return VectorStoreIndex.from_documents(docs, storage_context=StorageContext.from_defaults(vector_store=store))


def add_pdf(path):
    """Read one PDF and add its pages to the index. Returns the number of pages."""
    docs = SimpleDirectoryReader(input_files=[path]).load_data()
    index = load_index()
    for doc in docs:
        index.insert(doc)
    return len(docs)


def pdf_chunks(pdf):
    """Every chunk of one PDF, in page order, each starting with its label: [file.pdf, p. 12]."""
    load_index()  # makes sure the PDFs are indexed
    got = collection().get(where={"file_name": pdf}, include=["documents", "metadatas"])
    chunks = [(m.get("page_label", "?"), f"[{pdf}, p. {m.get('page_label', '?')}]\n{d}")
              for d, m in zip(got["documents"], got["metadatas"])]
    chunks.sort(key=lambda c: int(c[0]) if str(c[0]).isdigit() else 0)
    return [text for _, text in chunks]
