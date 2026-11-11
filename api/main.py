"""Grisz's HTTP API. Run: uvicorn api.main:app --reload, then open http://127.0.0.1:8000/docs"""
from uuid import uuid4

from celery.result import AsyncResult
from fastapi import FastAPI, HTTPException, UploadFile
from langgraph.types import Command
from pydantic import BaseModel, Field

from agents.graph import graph
from agents.quiz import Kind, quiz_graph
from engine.rag import CALLBACKS, DATA
from worker.tasks import add_module, app as celery

MAX_PDF_MB = 50

app = FastAPI(title="Grisz")


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=1000)  # caps how much text one request sends to Groq


class QuizRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=200)
    kind: Kind = "mc"
    n: int = Field(5, ge=1, le=20)


class Decision(BaseModel):
    approve: bool


def thread(quiz_id):
    return {"configurable": {"thread_id": quiz_id}, "callbacks": CALLBACKS}


@app.post("/ask")
def ask(q: Question):
    result = graph.invoke({"question": q.question}, {"callbacks": CALLBACKS})
    return {"route": result["route"], "answer": result["answer"], "sources": result["sources"]}


@app.post("/modules")
def upload_module(file: UploadFile):
    """Teacher: upload a PDF. A background worker reads and indexes it; check progress at /jobs/{id}."""
    name = (file.filename or "").replace("\\", "/").split("/")[-1]  # drop any folder part of the name
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files.")
    if (file.size or 0) > MAX_PDF_MB * 1024 * 1024:
        raise HTTPException(413, f"PDF is over {MAX_PDF_MB} MB.")
    path = DATA / name
    if path.exists():
        raise HTTPException(409, "A module with that file name is already uploaded.")
    path.write_bytes(file.file.read())
    return {"job": add_module.delay(str(path)).id, "file": name}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    """PENDING (still working, or unknown id), SUCCESS (with page count) or FAILURE."""
    job = AsyncResult(job_id, app=celery)
    return {"job": job_id, "state": job.state, "result": job.result if job.successful() else None}


@app.post("/quiz")
def make_quiz(r: QuizRequest):
    """Teacher: generate a quiz. It waits for review; problems lists anything the checker couldn't fix."""
    quiz_id = uuid4().hex
    s = quiz_graph.invoke(r.model_dump(), thread(quiz_id))
    return {"id": quiz_id, "status": "pending", "quiz": s["quiz"], "problems": s["problems"]}


@app.post("/quiz/{quiz_id}/review")
def review(quiz_id: str, d: Decision):
    """Teacher: approve or reject a pending quiz."""
    snap = quiz_graph.get_state(thread(quiz_id))
    if not snap.values:
        raise HTTPException(404, "No quiz with that id.")
    if snap.next != ("teacher",):
        raise HTTPException(409, "This quiz was already reviewed.")
    s = quiz_graph.invoke(Command(resume=d.approve), thread(quiz_id))
    return {"id": quiz_id, "status": s["status"]}


@app.get("/quiz/{quiz_id}")
def get_quiz(quiz_id: str):
    """Student: the quiz appears only after a teacher approves it."""
    s = quiz_graph.get_state(thread(quiz_id)).values
    if not s:
        raise HTTPException(404, "No quiz with that id.")
    status = s.get("status", "pending")
    return {"id": quiz_id, "status": status, "quiz": s["quiz"] if status == "approved" else None}
