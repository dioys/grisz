"""Phase 6: Planner agent. A study plan from today to the exam, built from the module pages that match the topics."""
import os
from datetime import date, timedelta

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel

from engine.rag import context, load_index

MAX_SESSIONS = 14


class Session(BaseModel):
    date: str         # YYYY-MM-DD, one of the given study dates
    focus: str        # what to study that day
    pages: list[str]  # where, as "file, p. X", copied from the module text
    task: str         # one concrete thing to do


class Plan(BaseModel):
    sessions: list[Session]
    note: str         # one blunt line from Grisz


planner = ChatPromptTemplate.from_messages([
    ("system", "You are Grisz, a blunt, sarcastic study bear who always actually helps. Build a study plan from the module text below. "
               "Use exactly the study dates given, one session per date, in order; the last session is a review. "
               "Each session's pages are labels copied from the module text, like 'file.pdf, p. 12'. "
               "Use only topics found in the module text. Roast procrastination, never the person.\n\nModule text:\n{context}"),
    ("human", "Exam on {exam}. Topics: {topics}. Study dates: {dates}."),
]) | ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")).with_structured_output(Plan, method="json_schema")


def study_dates(exam, today=None):
    """Up to 14 dates from today to the day before the exam, spread evenly."""
    today = today or date.today()
    days = (exam - today).days
    n = min(days, MAX_SESSIONS)
    if n == 1:
        return [today]
    return [today + timedelta(days=round(i * (days - 1) / (n - 1))) for i in range(n)]


def make_plan(exam, topics, callbacks=()):
    dates = ", ".join(d.isoformat() for d in study_dates(exam))
    plan = planner.invoke({"context": context(load_index(), topics, k=10), "exam": exam.isoformat(),
                           "topics": topics, "dates": dates}, {"callbacks": list(callbacks)})
    return plan.model_dump()
