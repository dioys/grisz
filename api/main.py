"""Grisz's HTTP API. Run: uvicorn api.main:app --reload, then open http://127.0.0.1:8000/docs"""
import hmac
import json
import os
import time
from collections import defaultdict, deque
from datetime import date
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
from agents.planner import make_plan
from agents.quiz import Kind, quiz_graph
from engine.rag import CALLBACKS, DATA, ROOT
from api import results
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
    if request.url.path.startswith(("/ask", "/quiz", "/modules", "/plan")):  # /jobs is left out: the page checks it every 3 s
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


class Answers(BaseModel):
    student: str = Field(min_length=1, max_length=60)  # the name is saved with the score
    answers: list[str] = []         # one per question, in order
    matches: dict[str, str] = {}    # matching questions: term -> chosen meaning


class PlanRequest(BaseModel):
    exam_date: date
    topics: str = Field(min_length=1, max_length=300)


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


def quiz_state(quiz_id):
    s = quiz_graph.get_state(thread(quiz_id)).values
    if not s:
        raise HTTPException(404, "No quiz with that id.")
    return s


@app.get("/quiz/{quiz_id}", dependencies=[ANYONE])
def get_quiz(quiz_id: str):
    """Student: the questions only (no answers), once a teacher has approved the quiz."""
    s = quiz_state(quiz_id)
    status = s.get("status", "pending")
    if status != "approved":
        return {"id": quiz_id, "status": status, "quiz": None}
    quiz = s["quiz"]
    return {"id": quiz_id, "status": status, "quiz": {
        "items": [{k: it[k] for k in ("type", "q", "choices", "hint")} for it in quiz["items"]],
        "terms": [p["term"] for p in quiz["pairs"]],
        "meanings": sorted(p["meaning"] for p in quiz["pairs"])}}


@app.post("/quiz/{quiz_id}/grade", dependencies=[ANYONE])
def grade(quiz_id: str, a: Answers):
    """Student: check answers on the server, save the score with the student's name, return feedback."""
    s = quiz_state(quiz_id)
    if s.get("status") != "approved":
        raise HTTPException(409, "This quiz isn't open yet.")
    feedback, missed = [], []
    for i, it in enumerate(s["quiz"]["items"]):
        given = a.answers[i] if i < len(a.answers) else ""
        right = given.strip().lower() == it["answer"].strip().lower()
        feedback.append({"right": right, "answer": it["answer"], "why": it["why"], "wrong": it["wrong"].get(given, "")})
        if not right:
            missed.append(it["q"])
    for p in s["quiz"]["pairs"]:
        right = a.matches.get(p["term"]) == p["meaning"]
        feedback.append({"right": right, "answer": p["meaning"], "why": "", "wrong": ""})
        if not right:
            missed.append(p["term"])
    score = sum(f["right"] for f in feedback)
    results.save(quiz_id, a.student.strip(), score, len(feedback), missed)
    return {"score": score, "total": len(feedback), "feedback": feedback}


@app.get("/report", dependencies=[TEACHER])
def report():
    """Teacher: per quiz, every attempt (name, score, time), the average and the most-missed questions."""
    out = {}
    for quiz_id, student, score, total, missed, at in results.all_rows():
        q = out.setdefault(quiz_id, {"quiz_id": quiz_id, "attempts": [], "missed": {},
                                     "topic": quiz_graph.get_state(thread(quiz_id)).values.get("topic", "?")})
        q["attempts"].append({"student": student, "score": score, "total": total, "at": at})
        for label in json.loads(missed):
            q["missed"][label] = q["missed"].get(label, 0) + 1
    for q in out.values():
        q["average_percent"] = round(100 * sum(x["score"] for x in q["attempts"]) / max(1, sum(x["total"] for x in q["attempts"])))
        q["most_missed"] = sorted(q.pop("missed").items(), key=lambda kv: -kv[1])[:5]
    return {"quizzes": list(out.values())}


@app.delete("/results/{student}", dependencies=[TEACHER])
def delete_results(student: str):
    """Teacher: delete every saved result for one student (for deletion requests)."""
    return {"student": student, "deleted": results.delete(student)}


@app.post("/plan", dependencies=[ANYONE])
def plan(r: PlanRequest):
    """Student: a study plan from today to the exam, from the matching module pages."""
    if not 1 <= (r.exam_date - date.today()).days <= 180:
        raise HTTPException(400, "The exam date must be between tomorrow and 180 days from now.")
    return make_plan(r.exam_date, r.topics, CALLBACKS)


# The simple test page at http://127.0.0.1:8000/ (added last, so the API routes above win)
app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="page")
