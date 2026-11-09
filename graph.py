import os
from dotenv import load_dotenv

load_dotenv()

# LangSmith — must be set before importing LangChain/LangGraph
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"]   = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"]    = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]    = os.getenv("LANGCHAIN_PROJECT", "Blog_Agent")

from typing import Callable, ParamSpec, TypeVar, TypedDict
from langgraph.graph import StateGraph, END

from agent import (
    plan_blog,
    research,
    analyze_competitor_gap,
    extract_facts,
    write_blog,
    audit_facts,
    critique_and_rewrite,
    targeted_revision,
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
    gap: str
    research_data: str
    facts: str
    memory: str
    draft: str
    critique: str       # editorial critic output — read by quality router
    final_blog: str
    seo: dict
    extras: dict
    scores: dict

    iteration: int
    max_iterations: int


# ── Node wrapper — cancels the whole pipeline on any node failure ─────────────

P = ParamSpec("P")
R = TypeVar("R")


def _guarded(fn: Callable[P, R]) -> Callable[P, R]:
    """Decorator that cancels the pipeline if the node raises."""
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return fn(*args, **kwargs)
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
def node_research(state: BlogState) -> dict:
    # research() now returns (dossier, raw_facts)
    # gap is already computed in parallel by node_competitor_gap
    dossier, raw_facts = research(state["topic"], state.get("gap", ""))
    return {
        "research_data": dossier,
        "facts": raw_facts,
    }

@_guarded
def node_competitor_gap(state: BlogState) -> dict:
    return {"gap": analyze_competitor_gap(state["topic"])}

@_guarded
def node_extract_facts(state: BlogState) -> BlogState:
    # facts are already built by the research engine — this is a no-op pass-through
    state["facts"] = extract_facts(state["topic"], state["facts"])
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
        gap=state.get("gap", ""),
        facts=state["research_data"],  # dossier is the authoritative source
    )
    state["final_blog"] = state["draft"]
    state["iteration"] = 0
    return state

@_guarded
def node_audit_facts(state: BlogState) -> BlogState:
    """Strips hallucinated numbers — runs right after write, before critique."""
    state["final_blog"] = audit_facts(
        state["final_blog"], state["research_data"], state["topic"]
    )
    return state

@_guarded
def node_critique_rewrite(state: BlogState) -> BlogState:
    """Editorial Critic — produces structured critique, stores in state."""
    critique = critique_and_rewrite(
        state["final_blog"], state["topic"], state["audience"]
    )
    state["critique"] = critique
    state["iteration"] = state.get("iteration", 0) + 1
    return state

@_guarded
def node_targeted_revision(state: BlogState) -> BlogState:
    """Targeted Revision — fixes only issues flagged by the critic."""
    state["final_blog"] = targeted_revision(
        state["final_blog"], state["critique"], state["topic"], state["audience"]
    )
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
def node_citations(state: BlogState) -> dict[str, str]:
    return {
        "final_blog": generate_citations(
            state["final_blog"], state["facts"], state["topic"], state["facts"]
        )
    }

@_guarded
def node_extras(state: BlogState) -> dict[str, object]:
    return {"extras": generate_extras(state["final_blog"], state["topic"])}

@_guarded
def node_save_memory(state: BlogState) -> BlogState:
    store_memory(state["topic"])
    store_memory(state["final_blog"])
    return state


# ── Conditional edge: revision loop ─────────────────────────────────────────

def quality_router(state: BlogState) -> str:
    """
    Quality Router — pure Python, zero LLM call.
    Reads the Editorial Critic output to decide: pass or targeted revision.
    Max 1 revision pass to keep latency low.
    """
    critique = state.get("critique", "")
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 1)  # max 1 revision

    # Already revised once — move on regardless
    if iteration >= max_iter:
        return "done"

    # Check if critic said Pass: YES
    if "pass" in critique.lower():
        lines = [l.strip().lower() for l in critique.split("\n")]
        for line in lines:
            if line.startswith("pass") and "yes" in line:
                return "done"

    # Check overall scores — if critic flagged issues, revise
    import re
    scores = re.findall(r'score:\s*(\d+)/10', critique, re.IGNORECASE)
    if scores:
        low_scores = [int(s) for s in scores if int(s) < 7]
        if not low_scores:
            return "done"
        return "revise"

    # Default: if critique exists and no clear pass, do one revision
    if critique and iteration == 0:
        return "revise"
    return "done"


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(BlogState)

    g.add_node("planner",          node_plan)
    g.add_node("competitor_gap",   node_competitor_gap)
    g.add_node("research",         node_research)
    g.add_node("extract_facts",    node_extract_facts)
    g.add_node("retrieve_memory",  node_retrieve_memory)
    g.add_node("write",            node_write)
    g.add_node("audit_facts",      node_audit_facts)
    g.add_node("critic",           node_critique_rewrite)
    g.add_node("targeted_revision",node_targeted_revision)
    g.add_node("score",            node_score)
    g.add_node("fix_cliches",      node_fix_cliches)
    g.add_node("citations",        node_citations)
    g.add_node("extras",           node_extras)
    g.add_node("save_memory",      node_save_memory)

    g.set_entry_point("planner")

    g.add_edge("planner",          "competitor_gap")
    g.add_edge("competitor_gap",   "research")
    g.add_edge("research",         "extract_facts")
    g.add_edge("extract_facts",    "retrieve_memory")
    g.add_edge("retrieve_memory",  "write")
    g.add_edge("write",            "audit_facts")
    g.add_edge("audit_facts",      "critic")

    # Quality Router — pure Python, zero LLM
    g.add_conditional_edges(
        "critic",
        quality_router,
        {
            "revise": "targeted_revision",
            "done":   "score",
        }
    )

    # After targeted revision — go to score directly (no second critic loop)
    g.add_edge("targeted_revision", "score")

    g.add_edge("score",            "fix_cliches")
    g.add_edge("fix_cliches",      "citations")
    g.add_edge("fix_cliches",      "extras")
    g.add_edge(["citations", "extras"], "save_memory")
    g.add_edge("save_memory",      END)

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
        "gap":            "",
        "research_data":  "",
        "facts":          "",
        "memory":         "",
        "draft":          "",
        "critique":       "",
        "final_blog":     "",
        "seo":            {},
        "extras":         {},
        "scores":         {},
        "iteration":      0,
        "max_iterations": 1,  # max 1 revision for latency
    }

    try:
        result = blog_graph.invoke(initial_state)
    except Exception:
        cancel_pipeline()  # ensure flag is set so any background work stops
        raise

    return {
        "topic":      result["topic"],
        "audience":   result["audience"],
        "length":     result["length"],
        "plan":       result["plan"],
        "draft":      result["draft"],
        "final_blog": result["final_blog"],
        "extras":     result["extras"],
        "seo":        result["seo"],
        "scores":     result["scores"],
    }
