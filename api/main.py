"""Grisz's HTTP API. Run: uvicorn api.main:app --reload, then open http://127.0.0.1:8000/docs"""
import hmac
import os
import time
from collections import defaultdict, deque
from uuid import uuid4

from celery.result import AsyncResult
from fastapi import Depends, FastAPI, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from agents.graph import graph
from agents.quiz import Kind, quiz_graph
from engine.rag import CALLBACKS, DATA, ROOT
from worker.tasks import add_module, app as celery

MAX_PDF_MB = 50


def access(*roles):
    """Only lets a request through if its X-Access-Code header matches one of these roles' codes in .env."""
    def check(x_access_code: str = Header("")):
        codes = [os.getenv(f"{role.upper()}_CODE", "") for role in roles]
        if not any(codes):
            raise HTTPException(503, "Access codes are not set in .env.")  # fail closed: no code set = nobody gets in
        if not any(c and hmac.compare_digest(x_access_code.encode(), c.encode()) for c in codes):
            raise HTTPException(401, "Wrong or missing access code.")
    return Depends(check)


TEACHER = access("teacher")
ANYONE = access("teacher", "student")

app = FastAPI(title="Grisz")
Instrumentator().instrument(app).expose(app)  # request counts and timings at /metrics, for Prometheus

RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
recent = defaultdict(deque)  # IP address -> times of its recent requests


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    """Each IP gets RATE_LIMIT requests a minute: protects the Groq quota and slows down code guessing."""
    if request.url.path.startswith(("/ask", "/quiz", "/modules")):  # /jobs is left out: the page checks it every 3 s
        now, hits = time.monotonic(), recent[request.client.host if request.client else "?"]
        while hits and now - hits[0] > 60:
            hits.popleft()
        if len(hits) >= RATE_LIMIT:
            return JSONResponse({"detail": "Too many requests. Wait a minute."}, status_code=429)
        hits.append(now)
    return await call_next(request)
ROUTES = Counter("grisz_ask_routes", "Questions by router decision", ["route"])
QUIZ_TRIES = Histogram("grisz_quiz_tries", "Writer attempts per quiz (1 = right first time)", buckets=[1, 2, 3])


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


@app.post("/ask", dependencies=[ANYONE])
def ask(q: Question):
    result = graph.invoke({"question": q.question}, {"callbacks": CALLBACKS})
    ROUTES.labels(result["route"]).inc()
    return {"route": result["route"], "answer": result["answer"], "sources": result["sources"]}


@app.post("/modules", dependencies=[TEACHER])
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


@app.get("/jobs/{job_id}", dependencies=[TEACHER])
def job_status(job_id: str):
    """PENDING (still working, or unknown id), SUCCESS (with page count) or FAILURE."""
    job = AsyncResult(job_id, app=celery)
    return {"job": job_id, "state": job.state, "result": job.result if job.successful() else None}


@app.post("/quiz", dependencies=[TEACHER])
def make_quiz(r: QuizRequest):
    """Teacher: generate a quiz. It waits for review; problems lists anything the checker couldn't fix."""
    quiz_id = uuid4().hex
    s = quiz_graph.invoke(r.model_dump(), thread(quiz_id))
    QUIZ_TRIES.observe(s["tries"])
    return {"id": quiz_id, "status": "pending", "quiz": s["quiz"], "problems": s["problems"]}


@app.post("/quiz/{quiz_id}/review", dependencies=[TEACHER])
def review(quiz_id: str, d: Decision):
    """Teacher: approve or reject a pending quiz."""
    snap = quiz_graph.get_state(thread(quiz_id))
    if not snap.values:
        raise HTTPException(404, "No quiz with that id.")
    if snap.next != ("teacher",):
        raise HTTPException(409, "This quiz was already reviewed.")
    s = quiz_graph.invoke(Command(resume=d.approve), thread(quiz_id))
    return {"id": quiz_id, "status": s["status"]}


@app.get("/quiz/{quiz_id}", dependencies=[ANYONE])
def get_quiz(quiz_id: str):
    """Student: the quiz appears only after a teacher approves it."""
    s = quiz_graph.get_state(thread(quiz_id)).values
    if not s:
        raise HTTPException(404, "No quiz with that id.")
    status = s.get("status", "pending")
    return {"id": quiz_id, "status": status, "quiz": s["quiz"] if status == "approved" else None}


# The simple test page at http://127.0.0.1:8000/ (added last, so the API routes above win)
app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="page")
