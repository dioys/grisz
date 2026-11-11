"""Week 3: Quiz agent → Checker (code, then AI) → up to 2 retries → teacher approval."""
import os
import sqlite3
from typing import Literal, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel

from engine.rag import STORAGE, context, load_index

MAX_TRIES = 3  # first attempt + 2 retries
Kind = Literal["mc", "tf", "fill", "id", "matching", "mixed"]


# Gamy's quiz JSON
class Item(BaseModel):
    type: Literal["mc", "tf", "fill", "id"]
    q: str
    choices: list[str] = []     # mc only: 4 options
    answer: str
    why: str                    # one sentence
    hint: str
    wrong: dict[str, str] = {}  # mc: wrong option → what it actually is


class Pair(BaseModel):
    term: str
    meaning: str


class Quiz(BaseModel):
    items: list[Item] = []
    pairs: list[Pair] = []      # matching only


class Review(BaseModel):
    problems: list[str]         # empty = every answer is right


class State(TypedDict, total=False):
    topic: str
    kind: str
    n: int
    context: str
    quiz: dict
    problems: list[str]
    tries: int
    status: str


llm = ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))

writer = ChatPromptTemplate.from_messages([
    ("system", "You are Grisz, writing a quiz for adult students from the module text below. Use only facts from it. "
               "Ask only about the lesson: never course codes, semesters, grading or anyone's contact info.\n"
               "Types: mc = exactly 4 choices, answer copied exactly from the choices, 'wrong' explains each wrong choice; "
               "tf = answer is True or False; fill = the question has a ___ blank; id = the answer is a term. "
               "'why' is one sentence; 'hint' is a clue that doesn't give the answer away. "
               "matching = fill 'pairs' instead of 'items'. mixed = a mix of mc, tf, fill and id.\n\nModule text:\n{context}"),
    ("human", "Write {n} {kind} questions about: {topic}{feedback}"),
]) | llm.with_structured_output(Quiz)

checker = ChatPromptTemplate.from_messages([
    ("system", "You check a quiz against the module text below. List every question whose answer is wrong, "
               "not supported by the module text, or ambiguous, as 'Q<number>: <what is wrong>'. "
               "Return an empty list if all are correct.\n\nModule text:\n{context}"),
    ("human", "{quiz}"),
]) | llm.with_structured_output(Review, method="json_schema")  # JSON mode: always answers, even with no problems


def shape_problems(quiz, kind, n):
    """Free checks in code, before spending an AI call."""
    if kind == "matching":
        terms = [p.term.lower() for p in quiz.pairs]
        return ([f"Need {n} pairs, got {len(terms)}."] if len(terms) != n else []) + \
               (["Every term must be different."] if len(set(terms)) != len(terms) else [])
    out = [] if len(quiz.items) == n else [f"Need {n} questions, got {len(quiz.items)}."]
    for i, it in enumerate(quiz.items, 1):
        if kind != "mixed" and it.type != kind:
            out.append(f"Q{i}: type must be {kind}, not {it.type}.")
        if it.type == "mc" and (len(set(it.choices)) != 4 or it.answer not in it.choices):
            out.append(f"Q{i}: needs 4 different choices, and the answer must be one of them.")
        if it.type == "tf" and it.answer not in ("True", "False"):
            out.append(f"Q{i}: answer must be True or False.")
        if it.type == "fill" and "___" not in it.q:
            out.append(f"Q{i}: needs a ___ blank.")
    return out


def retrieve(s):
    return {"context": context(load_index(), s["topic"]), "tries": 0, "problems": []}


def write(s):
    feedback = "".join(f"\n- {p}" for p in s["problems"])
    if feedback:
        feedback = "\nYour last quiz had these problems. Fix them all:" + feedback
    quiz = writer.invoke({**s, "feedback": feedback})
    return {"quiz": quiz.model_dump(), "tries": s["tries"] + 1}


def check(s):
    quiz = Quiz(**s["quiz"])
    problems = shape_problems(quiz, s["kind"], s["n"]) or \
        checker.invoke({"context": s["context"], "quiz": quiz.model_dump_json()}).problems
    return {"problems": problems}


def teacher(s):
    approved = interrupt({"quiz": s["quiz"], "problems": s["problems"]})  # pauses here until /review
    return {"status": "approved" if approved else "rejected"}


g = StateGraph(State)
g.add_node("retrieve", retrieve)
g.add_node("write", write)
g.add_node("check", check)
g.add_node("teacher", teacher)
g.add_edge(START, "retrieve")
g.add_edge("retrieve", "write")
g.add_edge("write", "check")
g.add_conditional_edges("check", lambda s: "write" if s["problems"] and s["tries"] < MAX_TRIES else "teacher")
g.add_edge("teacher", END)
STORAGE.mkdir(exist_ok=True)
saver = SqliteSaver(sqlite3.connect(STORAGE / "quizzes.db", check_same_thread=False))  # pending quizzes survive restarts
quiz_graph = g.compile(checkpointer=saver)
