import os
from dotenv import load_dotenv

load_dotenv()

# LangSmith — must be set before importing LangChain/LangGraph
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"]   = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"]    = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]    = os.getenv("LANGCHAIN_PROJECT", "Blog_Agent")

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from agent import (
    plan_blog,
    research,
    extract_facts,
    write_blog,
    critique_and_rewrite,
    fix_cliches,
    generate_seo,
    generate_extras,
    generate_citations,
    score_blog,
    cancel_pipeline,
    reset_pipeline,
)
from memory import retrieve_memory, store_memory


# ── State ────────────────────────────────────────────────────────────────────

class BlogState(TypedDict):
    topic: str
    audience: str
    length: str

    plan: str
    research_data: str
    facts: str
    memory: str
    draft: str
    final_blog: str
    seo: dict
    extras: dict
    scores: dict

    iteration: int          # how many rewrite iterations so far
    max_iterations: int     # max allowed rewrites


# ── Node wrapper — cancels the whole pipeline on any node failure ─────────────

def _guarded(fn):
    """Decorator that cancels the pipeline if the node raises."""
    def wrapper(state: BlogState) -> BlogState:
        try:
            return fn(state)
        except Exception:
            cancel_pipeline()
            raise
    return wrapper


# ── Nodes ────────────────────────────────────────────────────────────────────

@_guarded
def node_plan(state: BlogState) -> BlogState:
    state["plan"] = plan_blog(state["topic"], state["audience"])
    return state

@_guarded
def node_research(state: BlogState) -> BlogState:
    state["research_data"] = research(state["topic"])
    return state

@_guarded
def node_extract_facts(state: BlogState) -> BlogState:
    state["facts"] = extract_facts(state["topic"], state["research_data"])
    return state

@_guarded
def node_retrieve_memory(state: BlogState) -> BlogState:
    state["memory"] = retrieve_memory(state["topic"])
    return state

@_guarded
def node_write(state: BlogState) -> BlogState:
    state["draft"] = write_blog(
        topic=state["topic"],
        audience=state["audience"],
        plan=state["plan"],
        research_data=state["research_data"],
        memory=state["memory"],
        length=state["length"],
        gap="",
    )
    state["final_blog"] = state["draft"]
    state["iteration"] = 0
    return state

@_guarded
def node_critique_rewrite(state: BlogState) -> BlogState:
    state["final_blog"] = critique_and_rewrite(
        state["final_blog"], state["topic"], state["audience"]
    )
    state["iteration"] = state.get("iteration", 0) + 1
    return state

@_guarded
def node_score(state: BlogState) -> BlogState:
    seo = generate_seo(state["topic"], state["final_blog"])
    state["seo"] = seo
    state["scores"] = score_blog(state["final_blog"], state["topic"], seo)
    return state

@_guarded
def node_fix_cliches(state: BlogState) -> BlogState:
    state["final_blog"] = fix_cliches(state["final_blog"], state["topic"])
    return state

@_guarded
def node_citations(state: BlogState) -> BlogState:
    state["final_blog"] = generate_citations(
        state["final_blog"], state["facts"], state["topic"]
    )
    return state

@_guarded
def node_extras(state: BlogState) -> BlogState:
    state["extras"] = generate_extras(state["final_blog"], state["topic"])
    return state

@_guarded
def node_save_memory(state: BlogState) -> BlogState:
    store_memory(state["topic"])
    store_memory(state["final_blog"])
    return state


# ── Conditional edge: revision loop ─────────────────────────────────────────

def should_rewrite(state: BlogState) -> str:
    """
    After scoring, decide whether to rewrite again.
    If overall score < 8 and we haven't hit max iterations, rewrite.
    Otherwise move on.
    """
    overall_raw = state["scores"].get("Overall Score", "8")
    try:
        score = float(str(overall_raw).split()[0])
    except (ValueError, IndexError):
        score = 8.0

    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)

    if score < 8.0 and iteration < max_iter:
        return "rewrite"
    return "done"


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(BlogState)

    g.add_node("planner",         node_plan)
    g.add_node("research",        node_research)
    g.add_node("extract_facts",   node_extract_facts)
    g.add_node("retrieve_memory", node_retrieve_memory)
    g.add_node("write",           node_write)
    g.add_node("critique_rewrite",node_critique_rewrite)
    g.add_node("score",           node_score)
    g.add_node("fix_cliches",     node_fix_cliches)
    g.add_node("citations",       node_citations)
    g.add_node("extras",          node_extras)
    g.add_node("save_memory",     node_save_memory)

    g.set_entry_point("planner")

    g.add_edge("planner",         "research")
    g.add_edge("research",        "extract_facts")
    g.add_edge("extract_facts",   "retrieve_memory")
    g.add_edge("retrieve_memory", "write")
    g.add_edge("write",           "critique_rewrite")
    g.add_edge("critique_rewrite","score")

    # Conditional: rewrite again if score < 8, else continue
    g.add_conditional_edges(
        "score",
        should_rewrite,
        {
            "rewrite": "critique_rewrite",
            "done":    "fix_cliches",
        }
    )

    g.add_edge("fix_cliches",     "citations")
    g.add_edge("citations",       "extras")
    g.add_edge("extras",          "save_memory")
    g.add_edge("save_memory",     END)

    return g.compile()


# ── Public entry point ────────────────────────────────────────────────────────

blog_graph = build_graph()


def run_blog_pipeline(topic: str, audience: str, length: str = "medium") -> dict:
    reset_pipeline()  # clear any previous cancel state before starting

    initial_state: BlogState = {
        "topic":          topic,
        "audience":       audience,
        "length":         length,
        "plan":           "",
        "research_data":  "",
        "facts":          "",
        "memory":         "",
        "draft":          "",
        "final_blog":     "",
        "seo":            {},
        "extras":         {},
        "scores":         {},
        "iteration":      0,
        "max_iterations": 3,
    }

    try:
        result = blog_graph.invoke(initial_state)
    except Exception:
        cancel_pipeline()  # ensure flag is set so any background work stops
        raise

    return {
        "topic":      result["topic"],
        "audience":   result["audience"],
        "plan":       result["plan"],
        "draft":      result["draft"],
        "final_blog": result["final_blog"],
        "extras":     result["extras"],
        "seo":        result["seo"],
        "scores":     result["scores"],
    }
