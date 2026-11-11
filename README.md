# Grisz 🐻

A multi-agent study assistant. Upload course modules (PDFs) and Grisz, a blunt, roasting study bear:

- answers questions across all modules **with page citations**
- refuses off-topic questions (course codes, grades, contact info)
- writes quizzes, **checks them for mistakes and rewrites bad ones**, then waits for **teacher approval** before students see them
- builds a **study plan** up to the exam date and gives teachers a **progress report**

## Stack
LlamaIndex (PDF reading, retrieval, citations) · LangGraph (router, quiz → checker retry loop, human-in-the-loop) · LangChain (prompts, structured output) · FastAPI · Celery + Redis (background PDF indexing) · Chroma · Docker Compose · GitHub Actions · Groq (LLM) · fastembed (local embeddings)

```
POST /ask ──→ router ─(lesson)→ Q&A agent ──→ cited answer
                     └(off_topic)→ refusal
POST /quiz ─→ retrieve → write → check ─(problems)→ write (max 2 retries)
                                      └(ok)→ teacher approval ⏸
POST /modules ─→ Redis ─→ Celery worker ─→ Chroma
```

## Run it
Needs Docker Desktop and a free [Groq](https://console.groq.com) API key.
```bash
cp .env.example .env    # add GROQ_API_KEY, TEACHER_CODE and STUDENT_CODE
mkdir data              # put PDF modules here, or upload them via POST /modules
docker compose up --build
```
Open http://127.0.0.1:8000/ (simple test page) or http://127.0.0.1:8000/docs. Every request needs the `X-Access-Code` header: the teacher code or the student code.

| Endpoint | Who | What |
|---|---|---|
| `POST /ask` | student | cited answer, or a refusal if off-topic |
| `POST /modules` | teacher | upload a PDF; indexed in the background |
| `GET /jobs/{id}` | teacher | upload progress |
| `POST /quiz` | teacher | generate and check a quiz (mc, tf, fill, id, matching, mixed) |
| `POST /quiz/{id}/review` | teacher | approve or reject |
| `GET /quiz/{id}` | student | the questions (no answers), once approved |
| `POST /quiz/{id}/grade` | student | checked on the server; score saved with the student's name |
| `POST /plan` | student | study plan from today to the exam, from the matching module pages |
| `GET /report` | teacher | scores per quiz, averages, most-missed questions |
| `DELETE /results/{name}` | teacher | delete one student's saved results |

⚠️ Access is two shared codes (teacher and student), not personal accounts, and there is no rate limiting yet. Run it locally only.

## Test content
Developed with *Introduction to Python Programming* (OpenStax) and *AI for Teachers: an Open Textbook* (de la Higuera & Iyer), both CC BY 4.0. Module PDFs are not included in this repo.

## Licence
MIT, see [LICENSE](LICENSE).
