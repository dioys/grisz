"""Background jobs (slow work the API hands off). Run in Docker: celery -A worker.tasks worker"""
import os

from celery import Celery

from agents.exam import write_exam
from api.db import query
from engine.rag import add_pdf, pdf_chunks

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
app = Celery("grisz", broker=REDIS_URL, backend=REDIS_URL)


@app.task
def add_module(path):
    return {"pages": add_pdf(path)}


@app.task
def make_exam(exam_id):
    """Writes every question for one exam, saving progress so the teacher's dashboard can show it."""
    exam = query("SELECT pdf, mix FROM exams WHERE id = ?", exam_id)[0]
    try:
        chunks = pdf_chunks(exam["pdf"])
        if not chunks:
            raise ValueError("No text found in this PDF. Is it a scanned image?")
        progress = lambda done, total: query("UPDATE exams SET progress = ? WHERE id = ?", f"{done}/{total} parts written", exam_id)
        questions, problems = write_exam(chunks, exam["mix"], progress)
        query("UPDATE exams SET status = 'review', questions = ?, problems = ? WHERE id = ?", questions, problems, exam_id)
    except Exception as e:
        query("UPDATE exams SET status = 'failed', problems = ? WHERE id = ?", [f"Writing failed: {e}"], exam_id)
        raise
