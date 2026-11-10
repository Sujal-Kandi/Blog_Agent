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
    import logging as _logging
    _log = _logging.getLogger("blog_agent")

    min_words = {"short": 500, "medium": 900, "long": 1800}
    threshold = min_words.get(state["length"], 900)

    research_data = state["research_data"]
    facts = state.get("facts", "")

    # Guard: if research produced nothing (0 claims, empty dossier) the writer
    # has no grounding and will either hallucinate or produce a near-empty blog.
    # Inject an explicit instruction so the writer knows to draw on general
    # knowledge rather than a dossier — it stays honest and still produces a
    # full-length blog (this is the Hinduism/philosophy case).
    if not research_data or len(research_data.strip()) < 100:
        _log.warning(
            "[WRITE] Research dossier is empty or near-empty — "
            "writer will use general knowledge with explicit disclosure."
        )
        research_data = (
            "NOTE: Web research returned no structured claims for this topic. "
            "Draw on well-established general knowledge. "
            "Do NOT invent statistics or numbers — write without them. "
            "Be clear and confident about what is widely known; "
            "attribute nothing to a specific source unless you are certain."
        )
        # Also clear facts so citations node doesn't append a broken Sources block
        facts = facts or ""

    draft = ""
    for attempt in range(3):
        draft = write_blog(
            topic=state["topic"],
            audience=state["audience"],
            plan=state["plan"],
            research_data=research_data,
            memory=state["memory"],
            length=state["length"],
            gap=state.get("gap", ""),
            facts=facts if facts else research_data,
        )
        word_count = len(draft.split())
        if word_count >= threshold:
            break
        _log.warning(
            f"[WRITE] Output too short ({word_count} words, need {threshold}), retrying ({attempt + 1}/3)"
        )

    # Hard-fail guard: if after 3 attempts the draft is still suspiciously short,
    # log it loudly but keep whatever we got — better than a crash.
    if len(draft.split()) < threshold * 0.5:
        _log.error(
            f"[WRITE] Draft still under 50% of target after 3 attempts "
            f"({len(draft.split())} words). Keeping best effort."
        )

    state["draft"] = draft
    state["final_blog"] = draft
    state["iteration"] = 0
    return state

@_guarded
def node_audit_facts(state: BlogState) -> BlogState:
    """
    Strips hallucinated numbers — runs right after write, before critique.
    Skipped when research_data is a fallback stub (no real dossier available)
    to prevent the auditor from stripping the entire blog.
    """
    import logging as _logging
    _log = _logging.getLogger("blog_agent")

    research_data = state["research_data"]

    # If the dossier is the fallback stub injected by node_write, there are no
    # real verified facts to audit against — skip to avoid wiping the blog.
    FALLBACK_MARKER = "NOTE: Web research returned no structured claims"
    RAW_SNIPPET_MARKER = "── RAW SEARCH SNIPPETS"
    if (
        not research_data
        or FALLBACK_MARKER in research_data
        or RAW_SNIPPET_MARKER in research_data
        or len(research_data.strip()) < 150
    ):
        _log.warning(
            "[AUDIT] Skipping fact audit — dossier is a fallback stub, "
            "no verified facts to audit against."
        )
        return state

    before_words = len(state["final_blog"].split())
    audited = audit_facts(state["final_blog"], research_data, state["topic"])
    after_words = len(audited.split())

    # Safety: if audit wiped more than 40% of the blog, it over-pruned — revert
    if after_words < before_words * 0.6:
        _log.warning(
            f"[AUDIT] Audit removed too much content ({before_words} → {after_words} words, "
            f"{100*(1-after_words/before_words):.0f}% loss) — reverting to pre-audit blog"
        )
        return state

    state["final_blog"] = audited
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
def node_post_revision_check(state: BlogState) -> BlogState:
    """
    Post-Revision Safety Check — pure Python, zero LLM call.
    1. If the revision produced something shorter than the original draft
       (e.g. LLM truncated output), fall back to the original draft.
    2. Strips embedded critic/score blocks that sometimes leak into final_blog
       when the LLM returns critique text instead of a revised blog.
    3. Enforces the same minimum word count as node_write.
    """
    import re as _re
    import logging as _logging
    _log = _logging.getLogger("blog_agent")

    min_words = {"short": 500, "medium": 900, "long": 1800}
    threshold = min_words.get(state["length"], 900)

    revised_words = len(state["final_blog"].split())
    draft_words = len(state["draft"].split())

    if revised_words < threshold:
        _log.warning(
            f"[POST-REVISION] Revised blog too short ({revised_words} words, "
            f"need {threshold}) — falling back to original draft ({draft_words} words)"
        )
        state["final_blog"] = state["draft"]
    elif revised_words < draft_words * 0.75:
        # Revision lost more than 25% of content — treat as truncation, revert
        _log.warning(
            f"[POST-REVISION] Revision lost too much content "
            f"({draft_words} → {revised_words} words) — falling back to original draft"
        )
        state["final_blog"] = state["draft"]
    else:
        _log.info(f"[POST-REVISION] Revision passed check ({revised_words} words)")

    # ── Strip embedded critic output from final_blog ──────────────────────────
    # If the LLM returned the critique/score block instead of (or appended to)
    # the blog, strip everything from the first "DIMENSION" or "Overall Score:"
    # or "Overall Verdict" line onward, then re-check length.
    blog = state["final_blog"]
    critic_markers = [
        r'^DIMENSION\s+\d+', r'^Overall\s+(Score|Verdict)\s*[:\-]',
        r'^Pass\s*[:\-]', r'^Priority\s+Fix\s*[:\-]',
        r'^Blog Score\s*:', r'^## Blog Score',
    ]
    for marker in critic_markers:
        m = _re.search(marker, blog, _re.IGNORECASE | _re.MULTILINE)
        if m:
            truncated = blog[:m.start()].rstrip()
            if len(truncated.split()) >= threshold * 0.5:
                _log.warning(
                    f"[POST-REVISION] Stripped embedded critic block at pos {m.start()} "
                    f"(marker: {marker!r})"
                )
                state["final_blog"] = truncated
            break

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
    import logging as _logging
    _log = _logging.getLogger("blog_agent")

    original = state["final_blog"]
    facts = state.get("facts", "")

    try:
        result = generate_citations(
            original, facts, state["topic"], facts
        )
    except Exception as e:
        _log.error(f"[CITATIONS] generate_citations raised: {e} — keeping original blog")
        result = original

    # Safety: if citations somehow returned an empty/shorter-than-half string,
    # something went wrong — keep the original blog body intact.
    if not result or len(result.strip()) < len(original.strip()) * 0.5:
        _log.warning(
            "[CITATIONS] Result suspiciously short after citations — reverting to original"
        )
        result = original

    return {"final_blog": result}

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
    import re as _re
    critique = state.get("critique", "")
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 1)

    # Already at max iterations — move on regardless
    if iteration >= max_iter:
        return "done"

    # Explicit Pass: YES check — flexible regex, handles spacing/case variations
    pass_match = _re.search(r'pass\s*[:\-]?\s*(yes|no)', critique, _re.IGNORECASE)
    if pass_match:
        if pass_match.group(1).lower() == "yes":
            return "done"
        # Pass: NO — fall through to score check

    # Score check — revise only if at least one dimension is below 7
    scores = _re.findall(r'score:\s*(\d+)/10', critique, _re.IGNORECASE)
    if scores:
        low_scores = [int(s) for s in scores if int(s) < 7]
        return "revise" if low_scores else "done"

    # Critic returned no parseable scores (failed or empty) — skip revision,
    # don't waste a revision pass on garbage critique output
    return "done"


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(BlogState)

    g.add_node("planner",               node_plan)
    g.add_node("competitor_gap",         node_competitor_gap)
    g.add_node("research",               node_research)
    g.add_node("extract_facts",          node_extract_facts)
    g.add_node("retrieve_memory",        node_retrieve_memory)
    g.add_node("write",                  node_write)
    g.add_node("audit_facts",            node_audit_facts)
    g.add_node("critic",                 node_critique_rewrite)
    g.add_node("targeted_revision",      node_targeted_revision)
    g.add_node("post_revision_check",    node_post_revision_check)
    g.add_node("score",                  node_score)
    g.add_node("fix_cliches",            node_fix_cliches)
    g.add_node("citations",              node_citations)
    g.add_node("extras",                 node_extras)
    g.add_node("save_memory",            node_save_memory)

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

    # After targeted revision — safety check, then score
    g.add_edge("targeted_revision", "post_revision_check")
    g.add_edge("post_revision_check", "score")

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
