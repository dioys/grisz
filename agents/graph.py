"""Week 2: LangGraph router → Q&A agent, or a refusal for off-topic questions."""
import os
from typing import Literal, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from engine.rag import ask, load_index  # also loads .env

index = load_index()


class State(TypedDict, total=False):
    question: str
    route: str
    answer: str
    sources: list[str]


class Route(BaseModel):
    route: Literal["lesson", "off_topic"]


router = ChatPromptTemplate.from_messages([
    ("system", "Classify the student's message. 'lesson': about the study material, the modules themselves "
               "(title, authors, chapters, what they cover) or any school subject. "
               "'off_topic': course codes, semesters, grading, anyone's contact info, or nothing to do with studying. "
               "When unsure, choose 'lesson'."),
    ("human", "{question}"),
]) | ChatGroq(model=os.getenv("GROQ_ROUTER_MODEL", "openai/gpt-oss-20b")).with_structured_output(Route)


def route_question(state):
    return {"route": router.invoke({"question": state["question"]}).route}


def answer(state):
    return ask(index, state["question"])


def refuse(state):
    return {"answer": "Nice try. I only do your lesson, so ask me something from the modules.", "sources": []}


g = StateGraph(State)
g.add_node("router", route_question)
g.add_node("qa", answer)
g.add_node("off_topic", refuse)
g.add_edge(START, "router")
g.add_conditional_edges("router", lambda s: "qa" if s["route"] == "lesson" else "off_topic")
g.add_edge("qa", END)
g.add_edge("off_topic", END)
graph = g.compile()
