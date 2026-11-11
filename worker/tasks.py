"""Week 4: background jobs. Run in Docker: celery -A worker.tasks worker"""
import os

from celery import Celery

from engine.rag import add_pdf

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
app = Celery("grisz", broker=REDIS_URL, backend=REDIS_URL)


@app.task
def add_module(path):
    return {"pages": add_pdf(path)}
