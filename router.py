
import json
from datetime import datetime
from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field

from specialist_models import load_and_generate

from langgraph.graph.message import add_messages
from typing import Annotated
from langgraph.checkpoint.memory import MemorySaver

# ── Registry ─────────────────────────────────────────────────────────────
# general-assistant is NOT here — it's a hardcoded, always-resident node,
# not a swappable specialist competing in the router's catalog.

model_registry = []
with open("model_registry.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        model_registry.append(json.loads(line.strip()))

REGISTRY = {spec["id"]: spec for spec in model_registry}


# ── State ────────────────────────────────────────────────────────────────

class HarnessState(TypedDict):
    messages: Annotated[list, add_messages]
    specialist_id: str

# ── Router (resident — fires on every turn) ─────────────────────────────

router_llm = ChatOllama(model="qwen3:1.7b", keep_alive="-1s")

class RouteDecision(BaseModel):
    specialist_id: str = Field(
        description=(
            f"One of: {', '.join(REGISTRY.keys())}, "
            f"or 'general' if none of those fit the query"
        )
    )

def router_node(state: HarnessState) -> dict:
    writer = get_stream_writer()
    catalog = "\n".join(f"- {sid}: {spec['description']}" for sid, spec in REGISTRY.items())
    history_text = "\n".join(f"{m.type}: {m.content}" for m in state["messages"][-6:])
    prompt = (
        f"Available specialists:\n{catalog}\n\n"
        f"Recent conversation:\n{history_text}\n\n"
        f"Pick the single best specialist_id for the latest user message. "
        f"Use 'general' for casual conversation, meta questions about this system, "
        f"or anything not clearly in a specialist's domain."
    )
    decision = router_llm.with_structured_output(RouteDecision).invoke(prompt)
    writer({"type": "routing", "specialist_id": decision.specialist_id})
    return {"specialist_id": decision.specialist_id}

def route_decision(state: HarnessState) -> str:
    return "general" if state["specialist_id"] == "general" else "specialist"

# ── Specialist (on-demand, evicts quickly) ──────────────────────────────

def specialist_node(state: HarnessState) -> dict:
    writer = get_stream_writer()
    spec = REGISTRY[state["specialist_id"]]
    query = state["messages"][-1].content

    history_text = "\n".join(f"{m.type}: {m.content}" for m in state["messages"][-6:])
    combined_prompt = f"Conversation so far:\n{history_text}\n\nUser: {query}"

    full_text = ""
    for chunk in load_and_generate(combined_prompt, model_name=spec["ollama_tag"], keep_alive="5s"):
        writer({"type": "token", "content": chunk})
        full_text += chunk

    return {"messages": [{"role": "assistant", "content": full_text}]}

# ── General (permanently resident) ──────────────────────────────────────

GENERAL_SYSTEM_PROMPT = (
    "You are the general-purpose assistant inside a specialist-routing AI harness. "
    "Specific domains (code, legal questions, and others as they're added) are handled "
    "by dedicated fine-tuned specialist models, not you — you are only invoked when a "
    "query doesn't match any of them. Answer helpfully and conversationally within your "
    "own general knowledge. If a question is clearly better suited to a domain specialist "
    "that may exist, you can mention that, but you should still do your best to answer directly."
    f"You have the following experts at your desposal: {REGISTRY}"

)

general_llm = ChatOllama(model="qwen3:4b", keep_alive="-1s")
print("Warming up general assistant...")
list(general_llm.stream("hi"))
print("General assistant loaded.")

def general_node(state: HarnessState) -> dict:
    writer = get_stream_writer()
    query = state["messages"][-1].content
    combined_prompt = f"{GENERAL_SYSTEM_PROMPT}\n\nConversation so far:\n" + \
        "\n".join(f"{m.type}: {m.content}" for m in state["messages"][-6:]) + \
        f"\n\nUser: {query}"
    full_text = ""
    for chunk in general_llm.stream(combined_prompt):
        writer({"type": "token", "content": chunk.content})
        full_text += chunk.content
    return {"messages": [{"role": "assistant", "content": full_text}]}

# ── Graph ────────────────────────────────────────────────────────────────

graph = StateGraph(HarnessState)
graph.add_node("router", router_node)
graph.add_node("specialist", specialist_node)
graph.add_node("general", general_node)
graph.set_entry_point("router")
graph.add_conditional_edges("router", route_decision, {"specialist": "specialist", "general": "general"})
graph.add_edge("specialist", END)
graph.add_edge("general", END)

app = graph.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "session-1"}}

# ── Run loop ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    query = ""
    while query != "exit":
        query = input("Enter your query: ")
        if query == "exit":
            break

        st = datetime.now()
        for chunk in app.stream({"query": query}, stream_mode="custom"):
            print(chunk, end="", flush=True)
        et = datetime.now()

        print(f"\nTime taken: {et - st}")








#Open WebUI