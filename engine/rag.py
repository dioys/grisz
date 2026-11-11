"""Week 1: answer questions about module PDFs, with page citations, in Grisz's voice."""
import os
from functools import cache
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from llama_index.core import PromptTemplate, Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"        # put your module PDFs here
STORAGE = ROOT / "storage"  # vector DB + embedding model; delete it to rebuild

load_dotenv(ROOT / ".env")
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir=str(STORAGE / "models"))
Settings.llm = Groq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), api_key=os.environ["GROQ_API_KEY"])

GRISZ = PromptTemplate("""You are Grisz, a blunt, sarcastic talking bear who roasts lazy questions and wrong ideas but always actually helps.
Rules:
- Roast the question or the mistake, never the person: no insults about looks, identity or worth, no slurs. Never say that you are roasting.
- Answer only from the sources below. If the answer isn't there, say it's not in the module.
- Talk only about the lesson: never course codes, semesters, grading or anyone's contact info.
- Keep it under 120 words.
- Cite every fact with its source number in brackets, like [1] or [2].
- If the question has spelling or grammar mistakes, end with one line starting with "✏️" that corrects them. If there are none, leave the ✏️ line out.

Sources:
{context_str}

Question: {query_str}
Answer: """)


@cache  # load once, even when the Q&A and quiz agents both ask for it
def load_index():
    """Read the PDFs once and save their embeddings in Chroma; later runs reuse them."""
    host = os.getenv("CHROMA_HOST")  # set in Docker: the shared Chroma server; unset: local files
    client = chromadb.HttpClient(host=host, port=8000) if host else chromadb.PersistentClient(path=str(STORAGE / "chroma"))
    collection = client.get_or_create_collection("modules")
    store = ChromaVectorStore(chroma_collection=collection)
    if collection.count():
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


def ask(index, question):
    """Grisz's answer, plus the file and page behind each [n]."""
    engine = CitationQueryEngine.from_args(index, similarity_top_k=4, citation_qa_template=GRISZ)
    response = engine.query(question)
    sources = [f"[{i}] {n.metadata['file_name']}, p. {n.metadata['page_label']}"
               for i, n in enumerate(response.source_nodes, 1)]
    return {"answer": str(response), "sources": sources}


def context(index, topic, k=6):
    """The k module chunks closest to a topic, each labelled with its file and page."""
    nodes = index.as_retriever(similarity_top_k=k).retrieve(topic)
    return "\n\n".join(f"[{n.metadata['file_name']}, p. {n.metadata['page_label']}]\n{n.get_content()}"
                       for n in nodes)
