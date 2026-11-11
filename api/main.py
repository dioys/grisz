"""Grisz's HTTP API. Students: /exam/{code}. Teachers: /teacher/... (password). Pages: / and /teacher.html"""
import hmac
import os
import secrets
import time
from collections import Counter as Tally, defaultdict, deque
from uuid import uuid4

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from agents.exam import TYPES, grade
from api.db import query
from engine.rag import DATA, ROOT
from worker.tasks import add_module, app as celery, make_exam

MAX_PDF_MB = 50
CODE_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O or 1/I, so codes are easy to read out

app = FastAPI(title="Grisz")
Instrumentator().instrument(app).expose(app)  # request counts and timings at /metrics, for Prometheus
SUBMISSIONS = Counter("grisz_exam_submissions", "Exams handed in")
SCORES = Histogram("grisz_exam_score_percent", "Scores of handed-in exams, in percent", buckets=[25, 50, 75, 90, 100])

FAILED_LIMIT = int(os.getenv("FAILED_TRIES_PER_MINUTE", "10"))
failed = defaultdict(deque)  # IP address -> times of its recent wrong codes or passwords


@app.middleware("http")
async def slow_down_guessing(request: Request, call_next):
    """After 10 wrong exam codes or teacher passwords in a minute, that IP has to wait.
    Only failures count, so a whole class behind one school IP can still take an exam."""
    ip = request.client.host if request.client else "?"
    now, hits = time.monotonic(), failed[ip]
    while hits and now - hits[0] > 60:
        hits.popleft()
    if len(hits) >= FAILED_LIMIT:
        return JSONResponse({"detail": "Too many wrong tries. Wait a minute."}, status_code=429)
    response = await call_next(request)
    if response.status_code in (401, 404) and request.url.path.startswith(("/exam/", "/teacher/")):
        hits.append(now)
    return response


def teacher_password(x_teacher_password: str = Header("")):
    real = os.getenv("TEACHER_PASSWORD", "")
    if not real:
        raise HTTPException(503, "TEACHER_PASSWORD is not set in .env.")  # fail closed
    if not hmac.compare_digest(x_teacher_password.encode(), real.encode()):
        raise HTTPException(401, "Wrong password.")


teacher = APIRouter(prefix="/teacher", dependencies=[Depends(teacher_password)])


class Mix(BaseModel):
    tf: int = Field(0, ge=0, le=50)
    mc: int = Field(0, ge=0, le=50)
    id: int = Field(0, ge=0, le=50)
    fill: int = Field(0, ge=0, le=50)


class NewExam(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    pdf: str
    mix: Mix


class Mark(BaseModel):
    question: int = Field(ge=0)
    right: bool


class Submission(BaseModel):
    student: str = Field(min_length=1, max_length=60)  # saved with the score
    answers: list[str] = Field(max_length=200)


def get_exam(exam_id):
    rows = query("SELECT * FROM exams WHERE id = ?", exam_id)
    if not rows:
        raise HTTPException(404, "No exam with that id.")
    return rows[0]


# ---------- Teacher: lessons ----------

@teacher.get("/pdfs")
def list_pdfs():
    return sorted(p.name for p in DATA.glob("*.pdf"))


@teacher.post("/pdfs")
def upload_pdf(file: UploadFile):
    """Upload a lesson PDF. A background worker reads it; check progress at /teacher/jobs/{id}."""
    name = (file.filename or "").replace("\\", "/").split("/")[-1]  # drop any folder part of the name
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files.")
    if (file.size or 0) > MAX_PDF_MB * 1024 * 1024:
        raise HTTPException(413, f"PDF is over {MAX_PDF_MB} MB.")
    path = DATA / name
    if path.exists():
        raise HTTPException(409, "A lesson with that file name is already uploaded.")
    path.write_bytes(file.file.read())
    return {"job": add_module.delay(str(path)).id, "file": name}


@teacher.get("/jobs/{job_id}")
def job_status(job_id: str):
    """PENDING (still working, or unknown id), SUCCESS (with page count) or FAILURE."""
    job = AsyncResult(job_id, app=celery)
    return {"job": job_id, "state": job.state, "result": job.result if job.successful() else None}


# ---------- Teacher: exams ----------

@teacher.post("/exams")
def new_exam(e: NewExam):
    """Starts writing an exam in the background. Watch its status and progress in GET /teacher/exams."""
    if e.pdf not in list_pdfs():
        raise HTTPException(404, "No lesson PDF with that name.")
    mix = e.mix.model_dump()
    if not 1 <= sum(mix.values()) <= 100:
        raise HTTPException(400, "An exam needs 1 to 100 questions.")
    exam_id = uuid4().hex
    query("INSERT INTO exams (id, title, pdf, mix, status) VALUES (?, ?, ?, ?, 'writing')", exam_id, e.title, e.pdf, mix)
    make_exam.delay(exam_id)
    return {"id": exam_id}


@teacher.get("/exams")
def list_exams():
    return query("SELECT e.id, e.title, e.pdf, e.mix, e.status, e.progress, e.code, e.created, "
                 "(SELECT COUNT(*) FROM submissions s WHERE s.exam_id = e.id) AS submissions "
                 "FROM exams e ORDER BY e.created DESC")


@teacher.get("/exams/{exam_id}")
def review_exam(exam_id: str):
    """Every question with its answer, plus anything the checker couldn't fix."""
    return get_exam(exam_id)


@teacher.post("/exams/{exam_id}/publish")
def publish(exam_id: str):
    """Gives the exam a code that students type in to take it."""
    exam = get_exam(exam_id)
    if exam["status"] not in ("review", "closed"):
        raise HTTPException(409, f"An exam that is '{exam['status']}' can't be published.")
    code = exam["code"] or "".join(secrets.choice(CODE_LETTERS) for _ in range(6))
    query("UPDATE exams SET status = 'published', code = ? WHERE id = ?", code, exam_id)
    return {"code": code}


@teacher.post("/exams/{exam_id}/close")
def close(exam_id: str):
    """The code stops working; results are kept."""
    get_exam(exam_id)
    query("UPDATE exams SET status = 'closed' WHERE id = ?", exam_id)
    return {"status": "closed"}


@teacher.delete("/exams/{exam_id}")
def delete_exam(exam_id: str):
    """Deletes the exam and every student's result for it."""
    get_exam(exam_id)
    query("DELETE FROM submissions WHERE exam_id = ?", exam_id)
    query("DELETE FROM exams WHERE id = ?", exam_id)
    return {"deleted": exam_id}


# ---------- Teacher: grading ----------

@teacher.get("/exams/{exam_id}/results")
def results(exam_id: str):
    """Every submission (answers and marks), the average, and the 10 most-missed questions."""
    exam = get_exam(exam_id)
    subs = query("SELECT id, student, answers, marks, score, total, created FROM submissions WHERE exam_id = ? ORDER BY created", exam_id)
    missed = Tally(i for s in subs for i, ok in enumerate(s["marks"]) if not ok)
    return {
        "title": exam["title"],
        "questions": exam["questions"],
        "submissions": subs,
        "average_percent": round(100 * sum(s["score"] for s in subs) / max(1, sum(s["total"] for s in subs))),
        "most_missed": [{"question": i, "q": exam["questions"][i]["q"], "missed": n} for i, n in missed.most_common(10)],
    }


@teacher.post("/submissions/{submission_id}/mark")
def override(submission_id: int, m: Mark):
    """The teacher flips one question's mark; the score is recounted."""
    rows = query("SELECT marks FROM submissions WHERE id = ?", submission_id)
    if not rows:
        raise HTTPException(404, "No submission with that id.")
    marks = rows[0]["marks"]
    if m.question >= len(marks):
        raise HTTPException(400, "No question with that number.")
    marks[m.question] = m.right
    query("UPDATE submissions SET marks = ?, score = ? WHERE id = ?", marks, sum(marks), submission_id)
    return {"score": sum(marks), "total": len(marks)}


@teacher.delete("/students/{name}")
def delete_student(name: str):
    """Deletes every saved result for one student name (for deletion requests)."""
    n = query("SELECT COUNT(*) AS n FROM submissions WHERE lower(student) = lower(?)", name)[0]["n"]
    query("DELETE FROM submissions WHERE lower(student) = lower(?)", name)
    return {"student": name, "deleted": n}


app.include_router(teacher)


# ---------- Students ----------

def open_exam(code):
    rows = query("SELECT * FROM exams WHERE code = ? AND status = 'published'", code.strip().upper())
    if not rows:
        raise HTTPException(404, "No open exam with that code.")
    return rows[0]


@app.get("/exam/{code}")
def take_exam(code: str):
    """The questions only: no answers until the exam is handed in."""
    exam = open_exam(code)
    return {"title": exam["title"], "sections": TYPES,
            "questions": [{"type": q["type"], "q": q["q"], "choices": q["choices"]} for q in exam["questions"]]}


@app.post("/exam/{code}/submit")
def submit(code: str, s: Submission):
    """Grades the answers, saves the score with the student's name, and shows the answers with Grisz's notes."""
    exam = open_exam(code)
    questions = exam["questions"]
    marks = grade(questions, s.answers)
    score = sum(marks)
    query("INSERT INTO submissions (exam_id, student, answers, marks, score, total) VALUES (?, ?, ?, ?, ?, ?)",
          exam["id"], s.student.strip(), s.answers, marks, score, len(marks))
    SUBMISSIONS.inc()
    SCORES.observe(100 * score / max(1, len(marks)))
    return {"score": score, "total": len(marks), "results": [
        {"right": ok, "given": s.answers[i] if i < len(s.answers) else "", "answer": q["answer"],
         "why": q["why"], "wrong": q["wrong"].get(s.answers[i], "") if i < len(s.answers) else "", "page": q["page"]}
        for i, (q, ok) in enumerate(zip(questions, marks))]}


# The pages: / (students) and /teacher.html (teachers). Added last, so the API routes above win.
app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="pages")
