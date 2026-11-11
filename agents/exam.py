"""Exam agent: writes an exam from one lesson PDF in batches of up to 10 questions.
Each batch: Writer → Checker (code checks, then an AI fact-check) → up to 2 retries. Also grades typed answers."""
import json
import os
import re
from typing import Literal, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from engine.rag import CALLBACKS

TYPES = {"tf": "True or false", "mc": "Multiple choice", "id": "Identification", "fill": "Fill in the blank"}
BATCH = 10      # questions per AI call
MAX_TRIES = 3   # first attempt + 2 retries


class Item(BaseModel):  # Gamy's quiz JSON, plus the page the answer comes from
    type: Literal["tf", "mc", "id", "fill"]
    q: str
    choices: list[str] = []     # mc only: 4 options
    answer: str
    why: str                    # one sentence in Grisz's voice
    hint: str
    page: str                   # "file.pdf, p. 12"
    wrong: dict[str, str] = {}  # mc: wrong option → what it actually is


class Batch(BaseModel):
    items: list[Item]


class Review(BaseModel):
    problems: list[str]         # empty = every answer is right


class Marks(BaseModel):
    right: list[bool]


class State(TypedDict, total=False):
    context: str
    kind: str
    n: int
    batch: dict
    problems: list[str]
    tries: int


llm = ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), max_retries=6)  # retries wait out Groq's free limits

writer = ChatPromptTemplate.from_messages([
    ("system", "You are Grisz, a blunt, sarcastic study bear, writing exam questions for adult students from the lesson text below. "
               "Use only facts from it, and never ask about course codes, semesters, grading or anyone's contact info. "
               "Ask about what the lesson teaches, never about the book itself: not its title, authors, publisher, licence, "
               "ISBN, copyright, partners, table of contents or index.\n"
               "Types: tf = a statement whose answer is True or False; mc = exactly 4 different choices and the answer copied "
               "exactly from them, with 'wrong' explaining each wrong choice; id = identification, the answer is a short term "
               "(1 to 4 words); fill = the question has one ___ blank and the answer is the missing word or short phrase.\n"
               "'why' is one sentence in your voice explaining the answer: roast the mistake, never the person. "
               "'hint' doesn't give the answer away. 'page' is the label of the text the answer comes from, like 'file.pdf, p. 12'. "
               "No two questions about the same fact.\n\nLesson text:\n{context}"),
    ("human", "Write exactly {n} questions of type {kind}.{feedback}"),
]) | llm.with_structured_output(Batch, method="json_schema", include_raw=True)  # raw kept so we can repair it

checker = ChatPromptTemplate.from_messages([
    ("system", "You check exam questions against the lesson text below. List every question whose answer is wrong, "
               "not supported by the text, ambiguous, or about the book itself instead of the lesson (title, authors, "
               "publisher, licence, ISBN, table of contents), as 'Q<number>: <what is wrong>'. "
               "Return an empty list if all are correct.\n\nLesson text:\n{context}"),
    ("human", "{batch}"),
]) | llm.with_structured_output(Review, method="json_schema")

judge = ChatPromptTemplate.from_messages([
    ("system", "You grade typed exam answers. For each numbered item, decide if the student's answer means the same as the "
               "correct answer. Allow spelling slips, letter case and synonyms; a different or incomplete idea is wrong. "
               "Return exactly one true or false per item, in order."),
    ("human", "{items}"),
]) | ChatGroq(model=os.getenv("GROQ_JUDGE_MODEL", "openai/gpt-oss-20b"), max_retries=6).with_structured_output(Marks, method="json_schema")


def shape_problems(batch, kind, n):
    """Free checks in code, before spending an AI call."""
    out = [] if len(batch.items) == n else [f"Need {n} questions, got {len(batch.items)}."]
    for i, it in enumerate(batch.items, 1):
        if it.type != kind:
            out.append(f"Q{i}: type must be {kind}, not {it.type}.")
        if it.type == "mc" and (len(set(it.choices)) != 4 or it.answer not in it.choices):
            out.append(f"Q{i}: needs 4 different choices, and the answer must be one of them.")
        if it.type == "tf" and it.answer not in ("True", "False"):
            out.append(f"Q{i}: answer must be True or False.")
        if it.type == "fill" and "___" not in it.q:
            out.append(f"Q{i}: needs a ___ blank.")
    return out


def write(s):
    feedback = "".join(f"\n- {p}" for p in s["problems"])
    if feedback:
        feedback = "\nYour last attempt had these problems. Fix them all:" + feedback
    try:
        out = writer.invoke({**s, "feedback": feedback})
        batch = out["parsed"] or repair(out["raw"].content)
    except Exception:  # e.g. Groq rejected a malformed reply: an empty batch makes the checker ask for a rewrite
        batch = Batch(items=[])
    return {"batch": batch.model_dump(), "tries": s["tries"] + 1}


def repair(text):
    """The model sometimes wraps its list, e.g. {"items": {"items": [...]}}. Unwrap it; if that fails, return an
    empty batch, which the checker reports as a problem so the batch is written again."""
    try:
        data = json.loads(text)
        while isinstance(data.get("items"), dict):
            data = data["items"]
        return Batch(items=data["items"])
    except Exception:
        return Batch(items=[])


def check(s):
    batch = Batch(**s["batch"])
    problems = shape_problems(batch, s["kind"], s["n"]) or \
        checker.invoke({"context": s["context"], "batch": batch.model_dump_json()}).problems
    return {"problems": problems}


g = StateGraph(State)
g.add_node("write", write)
g.add_node("check", check)
g.add_edge(START, "write")
g.add_edge("write", "check")
g.add_conditional_edges("check", lambda s: "write" if s["problems"] and s["tries"] < MAX_TRIES else END)
batch_graph = g.compile()


def plan_batches(mix):
    """{"tf": 20, "mc": 15} → [("tf", 10), ("tf", 10), ("mc", 10), ("mc", 5)], in the order of TYPES."""
    return [(kind, min(BATCH, mix.get(kind, 0) - i)) for kind in TYPES for i in range(0, mix.get(kind, 0), BATCH)]


BOOK_INFO = re.compile(r"ISBN|Creative Commons|All rights reserved|Library of Congress", re.I)


def split_text(chunks, k, limit=6000):
    """Split a PDF's chunks (in page order) into k parts, one per batch, and pick chunks spread across each part
    (up to `limit` characters), so the questions cover the whole PDF, not just the first pages of each part.
    Copyright and licence pages are skipped."""
    chunks = [c for c in chunks if not BOOK_INFO.search(c)] or chunks
    parts = []
    for b in range(k):
        part = chunks[b * len(chunks) // k:(b + 1) * len(chunks) // k] or [chunks[b % len(chunks)]]
        take = max(1, min(len(part), int(limit // max(1, sum(map(len, part)) / len(part)))))
        picked = [part[round(i * (len(part) - 1) / (take - 1))] for i in range(take)] if take > 1 else [part[len(part) // 2]]
        parts.append("\n\n".join(dict.fromkeys(picked))[:limit])
    return parts


def write_exam(chunks, mix, progress=lambda done, total: None):
    """All questions for one exam, plus any problems the checker couldn't get fixed."""
    batches = plan_batches(mix)
    questions, problems = [], []
    for i, ((kind, n), text) in enumerate(zip(batches, split_text(chunks, len(batches)))):
        progress(i, len(batches))
        s = batch_graph.invoke({"context": text, "kind": kind, "n": n, "problems": [], "tries": 0}, {"callbacks": CALLBACKS})
        start = len(questions) + 1
        questions += s["batch"]["items"]
        problems += [f"{TYPES[kind]}, questions {start}–{len(questions)}: {p}" for p in s["problems"]]
    progress(len(batches), len(batches))
    return questions, problems


def grade(questions, answers):
    """One True/False mark per question. Exact matches are right; typed near-misses are judged by the AI."""
    marks, typed = [], []
    for i, q in enumerate(questions):
        given = answers[i].strip() if i < len(answers) else ""
        right = given.lower() == q["answer"].strip().lower()
        marks.append(right)
        if not right and given and q["type"] in ("id", "fill"):
            typed.append(i)
    if typed:
        items = "\n".join(f"{n}. Question: {questions[i]['q']}\n   Correct answer: {questions[i]['answer']}\n   Student's answer: {answers[i]}"
                          for n, i in enumerate(typed, 1))
        verdicts = judge.invoke({"items": items}, {"callbacks": CALLBACKS}).right
        if len(verdicts) == len(typed):  # otherwise they stay wrong; the teacher can override
            for i, ok in zip(typed, verdicts):
                marks[i] = ok
    return marks
