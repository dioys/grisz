# Grisz 🐻

An AI exam maker for teachers. A teacher uploads a lesson PDF, picks the mix (say 20 true or false, 15 multiple choice, 15 identification), and Grisz, a blunt, roasting study bear, writes the exam **from that PDF only**, checks every question, and rewrites the bad ones. The teacher reviews it, publishes it, and gets a short code. Students type the code and their name, take the exam, and get their score with the answers and Grisz's (roasting) explanations. The teacher's dashboard shows every result and can override any mark.

## Stack
LlamaIndex (PDF reading, embeddings) · LangGraph (writer → checker retry loop per batch) · LangChain (prompts, structured output) · Groq (LLM) · fastembed (local embeddings) · Chroma · FastAPI · Celery + Redis (background PDF reading and exam writing) · SQLite · Docker Compose · GitHub Actions · Prometheus + Grafana · Langfuse (optional tracing) · Caddy (production HTTPS)

```
Teacher: upload PDF ─→ Celery worker ─→ Chroma
Teacher: make exam  ─→ Celery worker: for each batch of ≤10 questions
                         writer ─→ checker (code checks, then AI fact-check) ─(problems, max 2 retries)→ writer
                       ─→ teacher reviews ─→ publish ─→ exam code
Student: code + name ─→ exam ─→ hand in ─→ graded (typed answers judged by AI) ─→ score + answers
Teacher: results, most-missed questions, override any mark
```

## Run it
Needs Docker Desktop and a free [Groq](https://console.groq.com) API key.
```bash
cp .env.example .env    # add GROQ_API_KEY and TEACHER_PASSWORD
docker compose up --build
```
- Teachers: http://127.0.0.1:8000/teacher.html
- Students: http://127.0.0.1:8000/ (with the code from the teacher)
- API docs: http://127.0.0.1:8000/docs · Dashboards: http://127.0.0.1:3000

## Privacy
Students' names, answers and scores are stored (SQLite, Docker volume `db`). Teachers can delete one student's results from the dashboard. Lesson text and typed answers are sent to Groq; to Langfuse too if tracing is on. Lesson PDFs are not included in this repo.

## Test content
Developed with *Introduction to Python Programming* (OpenStax) and *AI for Teachers: an Open Textbook* (de la Higuera & Iyer), both CC BY 4.0.

## Licence
MIT, see [LICENSE](LICENSE).
