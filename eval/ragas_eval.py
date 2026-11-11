"""Week 5: score Grisz's answers with Ragas. Run in the venv, from the Grisz folder: python -m eval.ragas_eval"""
import asyncio
import os

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")  # don't send usage data to Ragas

from llama_index.core import Settings
from openai import AsyncOpenAI
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

from engine.rag import ask, load_index  # also loads .env

QUESTIONS = [
    "What is machine learning?",
    "What is the difference between supervised and unsupervised learning?",
    "What should teachers watch out for when students use AI?",
    "What is a for loop used for?",
    "What does the input() function do?",
    "What is the capital of France?",  # not in the modules: Grisz should say so
]


class LocalEmbeddings(BaseRagasEmbedding):
    """Ragas reuses Grisz's own local embedding model: nothing new to download, nothing sent out."""

    def embed_text(self, text, **kwargs):
        return Settings.embed_model.get_text_embedding(text)

    async def aembed_text(self, text, **kwargs):
        return self.embed_text(text)


async def main():
    groq = AsyncOpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")
    judge = llm_factory(os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), provider="groq", client=groq)
    faithfulness = Faithfulness(llm=judge)                                # is every claim backed by the retrieved pages?
    relevancy = AnswerRelevancy(llm=judge, embeddings=LocalEmbeddings())  # does the answer address the question?
    index = load_index()
    scores = []
    print("faithful  relevant  question")
    for q in QUESTIONS:
        r = ask(index, q)
        f = await faithfulness.ascore(user_input=q, response=r["answer"], retrieved_contexts=r["contexts"])
        a = await relevancy.ascore(user_input=q, response=r["answer"])
        scores.append((f.value, a.value))
        print(f"{f.value:8.2f}  {a.value:8.2f}  {q}")
        await asyncio.sleep(5)  # stay under Groq's free per-minute limit
    print(f"{sum(s[0] for s in scores) / len(scores):8.2f}  {sum(s[1] for s in scores) / len(scores):8.2f}  AVERAGE (1.00 = best)")


asyncio.run(main())
