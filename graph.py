import os
from dotenv import load_dotenv

load_dotenv()

# LangSmith — must be set before importing LangChain/LangGraph
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"]   = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"]    = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]    = os.getenv("LANGCHAIN_PROJECT", "Blog_Agent")

from typing import Callable, ParamSpec, TypeVar, TypedDict, Any
from langgraph.graph import StateGraph, END

from agent import (
    plan_blog,
    research,
    analyze_competitor_gap,
    extract_facts,
    extract_insights,
    narrative_architect,
    generate_opening_hook,
    generate_ending,
    write_blog,
    audit_facts,
    forbidden_pattern_filter,
    grounding_score,
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
from research_engine import ClaimRegistry


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
    claim_registry: Any   # ClaimRegistry object from research_engine

    # new elite nodes
    insights: str
    hook: str
    narrative: str      # Narrative Architect beat-by-beat outline

    draft: str
    critique: str
    final_blog: str
    seo: dict
    extras: dict
    scores: dict
    filter_violations: list   # violations found by forbidden_pattern_filter
    grounding_report: dict    # per-section grounding scores

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
    # research() now returns (dossier, raw_facts, claim_registry)
    dossier, raw_facts, registry = research(state["topic"], state.get("gap", ""))
    return {
        "research_data": dossier,
        "facts": raw_facts,
        "claim_registry": registry,
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
def node_insights(state: BlogState) -> dict:
    """
    Insight Extractor — finds the 3 most non-obvious ideas hiding in the research.
    Gives the writer a real idea to build around instead of just facts.
    Runs in parallel with node_hook after memory retrieval.
    """
    result = extract_insights(
        state["topic"],
        state["research_data"],
        state["plan"],
    )
    return {"insights": result}


@_guarded
def node_narrative_architect(state: BlogState) -> dict:
    """
    Narrative Architect — designs the story shape before the writer starts.
    LLM reasons from first principles: what does this reader believe,
    what will surprise them, what emotional journey serves this topic.
    Outputs a beat-by-beat outline the writer follows as a contract.
    Runs after insights+hook, before write.
    """
    result = narrative_architect(
        topic=state["topic"],
        audience=state["audience"],
        insights=state.get("insights", ""),
        plan=state["plan"],
        length=state["length"],
    )
    return {"narrative": result}


@_guarded
def node_hook(state: BlogState) -> dict:
    """
    Opening Hook Generator — produces 5 candidate opening sentences, picks sharpest.
    The opening is the hardest sentence to write and the most important.
    Runs in parallel with node_insights after memory retrieval.
    """
    result = generate_opening_hook(
        state["topic"],
        state.get("insights", ""),
        state["plan"],
        state["audience"],
    )
    return {"hook": result}


@_guarded
def node_ending(state: BlogState) -> dict:
    """
    Ending Generator — generates 3 candidate endings after the draft exists.
    Replaces the writer's generic closing with a sharp, earned final paragraph.
    Runs after audit_facts, before citations.
    """
    import re as _re
    import logging as _logging
    _log = _logging.getLogger("blog_agent")

    draft = state["final_blog"]
    ending = generate_ending(draft, state["topic"], state.get("insights", ""), state.get("narrative", ""))

    if not ending or not ending.strip():
        _log.warning("[ENDING] Empty result — keeping original ending")
        return {"final_blog": draft}

    # Hard safety: strip any label artifacts that leaked through
    # These patterns should NEVER appear in the final blog
    label_pattern = _re.compile(
        r'^(ENDING\s+[A-C]|CHOSEN ENDING|FINAL TEXT|OPTION\s+[A-C]|---CHOSEN---|REFRAME:|CHALLENGE:|HARD TRUTH:).*$',
        _re.IGNORECASE | _re.MULTILINE
    )
    ending_clean = label_pattern.sub("", ending).strip()
    # Collapse multiple blank lines left by stripping
    ending_clean = _re.sub(r'\n{3,}', '\n\n', ending_clean).strip()

    if not ending_clean:
        _log.warning("[ENDING] Ending was all labels after stripping — keeping original")
        return {"final_blog": draft}

    # Strategy: find the last ### section heading and replace everything after it
    # with the new ending. If no headings, replace last 2 paragraphs.
    headings = list(_re.finditer(r'^###\s+.+', draft, _re.MULTILINE))
    if headings:
        last_heading = headings[-1]
        before_last_section = draft[:last_heading.start()]
        last_heading_text = last_heading.group(0)
        new_final_blog = before_last_section + last_heading_text + "\n\n" + ending_clean
    else:
        paragraphs = [p.strip() for p in draft.split("\n\n") if p.strip()]
        if len(paragraphs) > 2:
            new_final_blog = "\n\n".join(paragraphs[:-2]) + "\n\n" + ending_clean
        else:
            new_final_blog = draft

    # Safety: don't shrink the blog more than 15%
    if len(new_final_blog.split()) < len(draft.split()) * 0.85:
        _log.warning("[ENDING] Replacement too short — keeping original")
        return {"final_blog": draft}

    _log.info(f"[ENDING] Ending replaced successfully")
    return {"final_blog": new_final_blog}


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
            hook=state.get("hook", ""),
            insights=state.get("insights", ""),
            registry=state.get("claim_registry"),
            narrative=state.get("narrative", ""),
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

    # Strip any leaked internal labels from the final draft
    # (hook generator REASON:, ending generator labels, etc.)
    import re as _re2
    label_patterns = [
        r'^REASON\s*:.*$',
        r'^TYPE\s+\d+\s*[—\-].*:.*$',
        r'^CHOSEN\s+HOOK\s*:.*$',
        r'^ENDING\s+[A-C]\s*[—\-].*:?\s*$',
        r'^CHOSEN\s+ENDING\s*:.*$',
        r'^FINAL\s+TEXT\s*:.*$',
        r'^---CHOSEN---.*$',
    ]
    for pat in label_patterns:
        draft = _re2.sub(pat, '', draft, flags=_re2.IGNORECASE | _re2.MULTILINE)
    # Collapse multiple blank lines left by stripping
    draft = _re2.sub(r'\n{3,}', '\n\n', draft).strip()

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

    # Safety: if audit removed more than 20% of the blog, it over-pruned — revert.
    # The new surgical audit should only ever change a handful of sentences,
    # so even 20% is a very generous threshold.
    if after_words < before_words * 0.8:
        _log.warning(
            f"[AUDIT] Audit removed too much content ({before_words} → {after_words} words, "
            f"{100*(1-after_words/before_words):.0f}% loss) — reverting to pre-audit blog"
        )
        return state

    state["final_blog"] = audited
    return state

@_guarded
def node_forbidden_filter(state: BlogState) -> dict:
    """
    Phase 2 — Forbidden Pattern Filter.
    Pure Python + one targeted LLM call only for hard violations.
    Runs after audit_facts, before grounding_score.
    Structurally blocks: specific scores, chase targets, attributed quotes,
    named-event+scoreline combos that have no CLAIM-ID in the registry.
    """
    import logging as _logging
    _log = _logging.getLogger("blog_agent")

    registry = state.get("claim_registry")
    if not registry:
        _log.warning("[FILTER] No claim_registry in state — skipping forbidden filter")
        return {"filter_violations": []}

    # Skip on fallback dossiers
    research_data = state.get("research_data", "")
    if (
        not research_data
        or "NOTE: Web research returned no structured claims" in research_data
        or "── RAW SEARCH SNIPPETS" in research_data
    ):
        _log.warning("[FILTER] Fallback dossier — skipping forbidden filter")
        return {"filter_violations": []}

    filtered_blog, violations = forbidden_pattern_filter(
        state["final_blog"], registry, state["topic"]
    )

    # Safety: never shrink blog more than 10%
    before_w = len(state["final_blog"].split())
    after_w  = len(filtered_blog.split())
    if after_w < before_w * 0.90:
        _log.warning(f"[FILTER] Over-pruned ({before_w}→{after_w}) — reverting")
        return {"filter_violations": violations}

    return {"final_blog": filtered_blog, "filter_violations": violations}


@_guarded
def node_grounding_score(state: BlogState) -> dict:
    """
    Phase 3 — Grounding Score.
    Scores every ### section by grounded_sentences / total_specific_sentences.
    Sections below 0.60 get a targeted LLM rewrite constrained to registry values.
    Runs after forbidden_filter, before the ending generator.
    """
    import logging as _logging
    _log = _logging.getLogger("blog_agent")

    registry = state.get("claim_registry")
    if not registry:
        _log.warning("[GROUNDING] No claim_registry — skipping grounding score")
        return {"grounding_report": {}}

    research_data = state.get("research_data", "")
    if (
        not research_data
        or "NOTE: Web research returned no structured claims" in research_data
        or "── RAW SEARCH SNIPPETS" in research_data
    ):
        _log.warning("[GROUNDING] Fallback dossier — skipping grounding score")
        return {"grounding_report": {}}

    scored_blog, report = grounding_score(
        state["final_blog"], registry, state["topic"]
    )

    before_w = len(state["final_blog"].split())
    after_w  = len(scored_blog.split())
    if after_w < before_w * 0.80:
        _log.warning(f"[GROUNDING] Over-pruned ({before_w}→{after_w}) — reverting")
        return {"grounding_report": report}

    return {"final_blog": scored_blog, "grounding_report": report}


@_guarded
def node_critique_rewrite(state: BlogState) -> BlogState:
    """Story Critic — diagnoses narrative gaps, not just surface quality."""
    critique = critique_and_rewrite(
        state["final_blog"], state["topic"], state["audience"],
        state.get("narrative", "")
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
    return {"extras": generate_extras(state["final_blog"], state["topic"], state.get("facts", ""))}

@_guarded
def node_save_memory(state: BlogState) -> BlogState:
    store_memory(state["topic"])
    store_memory(state["final_blog"])
    return state


# ── Conditional edge: revision loop ─────────────────────────────────────────

def quality_router(state: BlogState) -> str:
    """
    Quality Router — pure Python, zero LLM call.
    Reads the Story Critic output to decide: pass or targeted revision.
    Max 1 revision pass to keep latency low.

    New story critic format: Pass: YES or NO
    Old rubric format fallback: scores.
    """
    import re as _re
    critique = state.get("critique", "")
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 1)

    if iteration >= max_iter:
        return "done"

    # Story critic format: "Pass: YES" or "Pass: NO"
    pass_match = _re.search(r'\bpass\s*[:\-]\s*(yes|no)\b', critique, _re.IGNORECASE)
    if pass_match:
        return "done" if pass_match.group(1).lower() == "yes" else "revise"

    # Fallback: old score format
    scores = _re.findall(r'score:\s*(\d+)/10', critique, _re.IGNORECASE)
    if scores:
        low_scores = [int(s) for s in scores if int(s) < 7]
        return "revise" if low_scores else "done"

    return "done"


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(BlogState)

    g.add_node("planner",               node_plan)
    g.add_node("competitor_gap",         node_competitor_gap)
    g.add_node("research",               node_research)
    g.add_node("extract_facts",          node_extract_facts)
    g.add_node("retrieve_memory",        node_retrieve_memory)
    g.add_node("insights",               node_insights)
    g.add_node("hook",                   node_hook)
    g.add_node("narrative_architect",    node_narrative_architect)   # Elite — story shape
    g.add_node("write",                  node_write)
    g.add_node("audit_facts",            node_audit_facts)
    g.add_node("forbidden_filter",       node_forbidden_filter)   # Phase 2 — NEW
    g.add_node("grounding_score",        node_grounding_score)    # Phase 3 — NEW
    g.add_node("ending",                 node_ending)
    g.add_node("critic",                 node_critique_rewrite)
    g.add_node("targeted_revision",      node_targeted_revision)
    g.add_node("post_revision_check",    node_post_revision_check)
    g.add_node("score",                  node_score)
    g.add_node("fix_cliches",            node_fix_cliches)
    g.add_node("citations",              node_citations)
    g.add_node("extras",                 node_extras)
    g.add_node("save_memory",            node_save_memory)

    g.set_entry_point("planner")

    # Plan → gap → research → extract_facts → memory
    g.add_edge("planner",          "competitor_gap")
    g.add_edge("competitor_gap",   "research")
    g.add_edge("research",         "extract_facts")
    g.add_edge("extract_facts",    "retrieve_memory")

    # After memory: insights + hook run IN PARALLEL
    g.add_edge("retrieve_memory",  "insights")
    g.add_edge("retrieve_memory",  "hook")

    # Both complete → Narrative Architect designs the story shape
    g.add_edge(["insights", "hook"], "narrative_architect")

    # Narrative Architect → Write (writer follows the beat outline)
    g.add_edge("narrative_architect", "write")

    g.add_edge("write",            "audit_facts")
    # After audit: forbidden filter blocks ungrounded specific claims (Phase 2)
    g.add_edge("audit_facts",      "forbidden_filter")
    # After filter: grounding score rewrites low-scoring sections (Phase 3)
    g.add_edge("forbidden_filter", "grounding_score")
    # After grounding: ending generator sharpens the closing paragraph
    g.add_edge("grounding_score",  "ending")
    g.add_edge("ending",           "critic")

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
    g.add_edge("targeted_revision",   "post_revision_check")
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
        "topic":            topic,
        "audience":         audience,
        "length":           length,
        "plan":             "",
        "gap":              "",
        "research_data":    "",
        "facts":            "",
        "memory":           "",
        "claim_registry":   None,
        "insights":         "",
        "hook":             "",
        "narrative":        "",
        "draft":            "",
        "critique":         "",
        "final_blog":       "",
        "seo":              {},
        "extras":           {},
        "scores":           {},
        "filter_violations": [],
        "grounding_report": {},
        "iteration":        0,
        "max_iterations":   1,
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
