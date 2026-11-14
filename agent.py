import os  #read env variables like API Keys
import re   # regex — used by forbidden_pattern_filter and grounding_score
import time #used for timing how long each step takes 
import uuid #generates unique IDs (imported but not actually used in this file)
import logging #writes logs to terminal and file 
import threading  # for the cancellation flag
from concurrent.futures import ThreadPoolExecutor, as_completed  #runs multiple tasks at the same time (parallel)
from dotenv import load_dotenv #reads your .env file and loadsthe keys into memory 



load_dotenv()



# ── Logging setup ─────────────────────────────────────────────────────────────
log_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("blog_agent")
logger.setLevel(logging.INFO)



# Console handler — shows in terminal and Render logs
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)



# File handler — saves to blog_agent.log locally
file_handler = logging.FileHandler("blog_agent.log")
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)


from langchain_groq import ChatGroq
from groq import AuthenticationError, RateLimitError
from tools import web_search
from langsmith import traceable
from research_engine import run_research_engine, ClaimRegistry


GROQ_KEYS = [
    value.strip()
    for name, value in sorted(os.environ.items())
    if (
        name == "GROQ_API_KEY"
        or name.startswith("GROQ_API_KEY_")
        or name.startswith("GROQ_KEY_")
    ) and value.strip()
]
_key_index = 0

# ── Cancellation flag ─────────────────────────────────────────────────────────
# Set this to True to stop all in-flight LLM calls and graph nodes immediately.
_cancel_event = threading.Event()


def cancel_pipeline():
    """Call this to hard-stop the current blog generation run."""
    _cancel_event.set()
    logger.warning("[CANCEL] Pipeline cancellation requested — stopping all steps.")


def reset_pipeline():
    """Call this before starting a new run to clear any previous cancel state."""
    _cancel_event.clear()


def _check_cancelled():
    """Raises RuntimeError immediately if a cancellation has been requested."""
    if _cancel_event.is_set():
        raise RuntimeError("Pipeline cancelled — stopping to avoid wasting tokens.")


def _invoke(prompt: str, temperature: float) -> str:
    global _key_index

    _check_cancelled()  # bail out before making any LLM call

    if not GROQ_KEYS:
        raise RuntimeError(
            "No Groq API keys configured. Set GROQ_API_KEY or one or more GROQ_API_KEY_* / GROQ_KEY_* variables."
        )

    authentication_failures = 0
    attempts = len(GROQ_KEYS) * 2
    for _ in range(attempts):
        _check_cancelled()
        try:
            llm = ChatGroq(
                model="openai/gpt-oss-120b",
                temperature=temperature,
                api_key=GROQ_KEYS[_key_index]
            )
            return llm.invoke(prompt).content
        except RuntimeError:
            raise  # propagate cancellation immediately
        except RateLimitError:
            print(f"Rate limit on Groq key {_key_index + 1}, switching...")
            _key_index = (_key_index + 1) % len(GROQ_KEYS)
            # Sleep in small chunks so cancellation is noticed quickly
            for _ in range(2):
                if _cancel_event.wait(timeout=1):
                    raise RuntimeError("Pipeline cancelled during Groq backoff.")
        except AuthenticationError:
            authentication_failures += 1
            print(f"Invalid Groq API key {_key_index + 1}, switching...")
            _key_index = (_key_index + 1) % len(GROQ_KEYS)

    if authentication_failures == attempts:
        raise RuntimeError(
            "All configured Groq API keys were rejected. Check the GROQ_API_KEY* / GROQ_KEY_* values in Render."
        )
    raise RuntimeError("All configured Groq API keys are rate-limited or invalid. Check Render logs and try again.")
#Groq Block



CLICHES = [
    "you can't have your cake and eat it too",
    "wake up and smell the coffee",
    "at the end of the day",
    "journey not a destination",
    "journey, not a destination",
    "humans + machines",
    "unlock your potential",
    "unlock a world of",
    "are you ready to",
    "start living",
    "game changer",
    "game-changer",
    "revolutionize",
    "in today's world",
    "in today's fast-paced",
    "have you ever wondered",
    "the future is now",
    "think outside the box",
    "it is what it is",
    "ticking time bomb",
    "warm handshake",
    "as the saying goes",
    "one size fits all",
    "paradigm shift",
    "move the needle",
    "circle back",
    "deep dive",
    "it goes without saying",
    "needless to say",
    "a perfect storm",
    "perfect storm",
    "culinary revolution",
    # ── Abstract dramatic phrases banned by Natural Language rule ─────────────
    # These sound impressive but replace concrete thinking with decoration.
    "acts as a crucible",
    "acted as a crucible",
    "as a crucible",
    "in the crucible",
    "the crucible of",
    "lighthouse in a storm",
    "shines like a lighthouse",
    "like a lighthouse",
    "a benediction",
    "turned that benediction",
    "transcends mere",
    "transcend mere",
    "the paradox of loss",
    "statistical fingerprint",
    "the weight of a single",
    "redefines greatness",
    "redefine greatness",
    "forged in the fire",
    "forged in defeat",
    "the anatomy of",
    "the architecture of greatness",
    "legacy reshaped",
    "reshaping his legacy",
    "the myth of the invincible",
    "hollow triumph",
    "a hollow victory",
    "tapestry of",
    "in the pantheon",
    "etched in history",
    "written in the annals",
    "the annals of",
    "stands as a testament",
    "testament to his",
]

LENGTH_GUIDE = {
    "short": "600 to 800 words. 3 to 4 sections. Every sentence must earn its place. Dense, sharp, zero filler. Short does not mean shallow — pack maximum insight into minimum words.",
    "medium": "1000 to 1500 words. 5 to 6 sections. Each section goes one level deeper than the surface. At least 2 specific data points or named examples per section.",
    "long": "2000 to 2500 words. 7 to 8 sections. Full depth on every angle. Multiple named examples, data points, and at least 2 strong analogies throughout. Every section should feel like it could stand alone."
}


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Forbidden Pattern Filter (pure Python, zero LLM, <0.05s)
# ═══════════════════════════════════════════════════════════════════════════════

# Patterns that are ABSOLUTELY FORBIDDEN unless backed by a registered CLAIM-ID.
# Each entry: (compiled_regex, human_label, severity)
# severity "hard" = sentence is rewritten by LLM to pattern language
# severity "soft" = flagged for grounding_score only, not auto-rewritten
_FORBIDDEN_PATTERNS = [
    # Specific chase/score targets: "needed 130 to win", "required 243 runs"
    (re.compile(r'\b(needed|required|chasing|chased)\s+\d[\d,]*\s+(runs?|to\s+win)', re.IGNORECASE),
     "specific_chase_target", "hard"),
    # Specific innings scores: "scored 91", "made 147 off 79 balls"
    (re.compile(r'\b(scored?|made|hit)\s+\d[\d,]*(?:\s+(?:runs?|not\s+out|off\s+\d+\s+balls?))?\b', re.IGNORECASE),
     "specific_innings_score", "hard"),
    # Match results: "won by 5 runs", "lost by 3 wickets"
    (re.compile(r'\b(won|lost)\s+by\s+\d[\d,]*\s+(runs?|wickets?)\b', re.IGNORECASE),
     "match_result", "hard"),
    # Specific over outcomes: "19-run over", "bowled a 2-run over"
    (re.compile(r'\b\d[\d]*[\s-]run\s+over\b', re.IGNORECASE),
     "over_outcome", "hard"),
    # Attributed direct quotes: 'Dhoni said "..."'
    (re.compile(
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+'
        r'(?:said|described|called|noted|wrote|told|explained|stated|admitted|revealed|claimed|added|replied)'
        r'\s+["\u201c\u2018][^"\u201d\u2019]{5,}["\u201d\u2019]',
        re.IGNORECASE),
     "attributed_quote", "hard"),
    # Named event + specific score in same sentence
    (re.compile(
        r'\b(?:In|During|At)\s+the\s+(?:\d{4}\s+)?'
        r'(?:IPL|World\s+Cup|Champions\s+Trophy|T20I?|ODI|Test|series|final|semi-final)'
        r'[^.!?]*\d[\d,]*\s*(?:runs?|wickets?|balls?|overs?)\b',
        re.IGNORECASE),
     "named_event_with_score", "hard"),
    # Fabricated percentages in performance context
    (re.compile(r'\b\d{1,3}\s*%\s+of\s+the\s+time\b', re.IGNORECASE),
     "fabricated_percentage", "soft"),
]


def forbidden_pattern_filter(
    blog: str,
    registry: "ClaimRegistry",
    topic: str,
) -> tuple[str, list[dict]]:
    """
    Phase 2 — Pure Python forbidden pattern filter.

    Scans every sentence for structurally unverifiable patterns.
    For each match, checks if the value is in the ClaimRegistry.
    If NOT grounded → queues sentence for LLM rewrite to pattern language.

    Returns:
        (filtered_blog, violations)
    """
    _check_cancelled()
    logger.info(f"[FILTER] Starting forbidden pattern filter | topic='{topic[:60]}'")
    start = time.time()

    grounded = registry.all_grounded_values()
    sentences = re.split(r'(?<=[.!?])\s+', blog)
    violations: list[dict] = []
    sentences_to_fix: list[tuple[int, str, str]] = []  # (idx, sentence, label)

    for idx, sentence in enumerate(sentences):
        s = sentence.strip()
        if not s or s.startswith("#"):
            continue

        for pattern, label, severity in _FORBIDDEN_PATTERNS:
            if not pattern.search(s):
                continue

            # Extract numbers in this sentence
            nums_in_sentence = {n.replace(",", "").lower() for n in re.findall(r'\b\d[\d,]*(?:\.\d+)?\b', s)}

            # Extract named event phrases
            event_phrases = re.findall(
                r'\b\d{4}\s+(?:IPL|World\s+Cup|Champions\s+Trophy|T20I?|ODI|Test|series|final|semi-final)\b',
                s, re.IGNORECASE
            )

            ungrounded_nums = nums_in_sentence - grounded
            ungrounded_events = [e for e in event_phrases if e.lower() not in grounded]

            # Attributed quotes are always flagged — quote text can't be in grounded values
            is_grounded = (
                label != "attributed_quote"
                and (severity == "soft" or not ungrounded_nums)
                and not ungrounded_events
            )

            if not is_grounded:
                violations.append({
                    "sentence": s,
                    "label": label,
                    "severity": severity,
                    "ungrounded_nums": list(ungrounded_nums),
                    "ungrounded_events": ungrounded_events,
                })
                if severity == "hard":
                    sentences_to_fix.append((idx, s, label))
            break  # one violation per sentence is enough

    if not violations:
        logger.info(f"[FILTER] Clean — no violations | duration={time.time()-start:.1f}s")
        return blog, []

    hard = [v for v in violations if v["severity"] == "hard"]
    soft = [v for v in violations if v["severity"] == "soft"]
    logger.info(f"[FILTER] {len(violations)} violations | hard={len(hard)} soft={len(soft)}")
    for v in violations:
        logger.info(f"  [{v['severity'].upper()}] [{v['label']}] {v['sentence'][:80]}")

    if not sentences_to_fix:
        return blog, violations

    # ── LLM rewrite — only hard violations ───────────────────────────────────
    sentences_block = "\n".join(f"- [{lbl}] {sent}" for _, sent, lbl in sentences_to_fix[:12])

    prompt = f"""
You are a fact-safety editor. The sentences below contain unverified specific claims
(match scores, chase targets, attributed quotes, specific over outcomes) that were NOT
found in the verified research dossier for this topic.

Topic: {topic}

Rewrite each sentence to remove the unverifiable specific detail while keeping the analytical point.

REWRITE RULES (use these as templates):
- "India needed 130 to win, Dhoni scored 91"       → "In that final, Dhoni anchored the chase with a measured, match-winning innings"
- "He scored 91 off 61 balls"                       → "He constructed his innings ball by ball, accelerating only when the equation demanded it"
- "won by 5 runs"                                   → "won in dramatic fashion"
- 'Dhoni said "I always stay calm"'                 → "Dhoni's actions in those moments spoke louder than any post-match comment"
- "In the 2013 CT final, India needed 130..."       → "In high-pressure knockout finals, Dhoni's approach..."

The rewritten sentence must:
1. Keep the ANALYTICAL POINT
2. Remove the SPECIFIC INVENTED DETAIL (exact score, exact quote, exact result)
3. Sound natural and confident — not hedged, not vague

Sentences to fix:
{sentences_block}

Format — one block per sentence:
ORIGINAL: [original sentence]
FIXED: [rewritten sentence]
"""

    try:
        raw = _invoke(prompt, temperature=0.0)
    except Exception as e:
        logger.warning(f"[FILTER] LLM rewrite failed: {e} — keeping original")
        return blog, violations

    fixed_blog = blog
    for block in raw.strip().split("\n\n"):
        lines_b = block.strip().split("\n")
        orig  = next((l[len("ORIGINAL:"):].strip() for l in lines_b if l.upper().startswith("ORIGINAL:")), None)
        fixed = next((l[len("FIXED:"):].strip()    for l in lines_b if l.upper().startswith("FIXED:")),    None)
        if orig and fixed and orig != fixed:
            fixed_blog = fixed_blog.replace(orig, fixed, 1)
            logger.info(f"[FILTER] Fixed: '{orig[:55]}' → '{fixed[:55]}'")

    logger.info(f"[FILTER] Done | duration={time.time()-start:.1f}s | rewrites={len(sentences_to_fix)}")
    return fixed_blog, violations


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Grounding Score (per-section, targeted LLM rewrite for low scorers)
# ═══════════════════════════════════════════════════════════════════════════════

def grounding_score(
    blog: str,
    registry: "ClaimRegistry",
    topic: str,
    threshold: float = 0.60,
) -> tuple[str, dict]:
    """
    Phase 3 — Per-section grounding score.

    For every ### section:
    1. Count sentences that contain specific claims (numbers, named events).
    2. Count how many have all their numbers present in the ClaimRegistry.
    3. score = grounded_sentences / total_specific_sentences
    4. Sections below threshold → targeted LLM rewrite.

    Returns:
        (scored_blog, report)   report = {section_heading: score}
    """
    _check_cancelled()
    logger.info(f"[GROUNDING] Starting grounding score | topic='{topic[:60]}'")
    start = time.time()

    grounded_vals = registry.all_grounded_values()

    # Split blog into (heading, body) pairs by ### headings
    section_pat = re.compile(r'(^###\s+.+$)', re.MULTILINE)
    parts = section_pat.split(blog)
    preamble = parts[0] if parts else ""

    sections: list[list[str]] = []   # [heading, body] — mutable for in-place rewrite
    i = 1
    while i < len(parts) - 1:
        sections.append([parts[i], parts[i + 1] if i + 1 < len(parts) else ""])
        i += 2

    if not sections:
        logger.info("[GROUNDING] No ### sections — skipping grounding score")
        return blog, {}

    # Pattern: sentence contains a specific claim (number or named event)
    specific_pat = re.compile(
        r'\b\d[\d,]*(?:\.\d+)?\s*(?:%|runs?|wickets?|matches?|balls?|overs?|years?|goals?|titles?)?\b'
        r'|\b(?:In|During|At)\s+the\s+(?:\d{4}\s+)?'
        r'(?:IPL|World\s+Cup|Champions\s+Trophy|T20I?|ODI|Test|final|semi-final)\b',
        re.IGNORECASE,
    )

    report: dict[str, float] = {}
    sections_to_rewrite: list[tuple[int, float]] = []  # (section_idx, score)

    for idx, (heading, body) in enumerate(sections):
        body_sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', body) if s.strip()]
        specific_sents = [s for s in body_sents if specific_pat.search(s)]

        if not specific_sents:
            report[heading.strip()] = 1.0
            continue

        grounded_count = sum(
            1 for s in specific_sents
            if not (
                {n.replace(",", "").lower() for n in re.findall(r'\b\d[\d,]*(?:\.\d+)?\b', s)}
                - grounded_vals
            )
        )
        score = grounded_count / len(specific_sents)
        report[heading.strip()] = round(score, 2)

        if score < threshold:
            sections_to_rewrite.append((idx, score))
            logger.info(f"[GROUNDING] Low-score section '{heading.strip()[:55]}' score={score:.2f}")

    logger.info(f"[GROUNDING] Scores: { {k[:40]: v for k, v in report.items()} }")

    if not sections_to_rewrite:
        logger.info(f"[GROUNDING] All sections pass — no rewrite | duration={time.time()-start:.1f}s")
        return blog, report

    registry_block = registry.to_writer_block()

    for idx, score in sections_to_rewrite:
        heading, body = sections[idx]
        logger.info(f"[GROUNDING] Rewriting '{heading.strip()[:55]}' (score={score:.2f})")

        prompt = f"""
You are a precise editor rewriting one section of a blog about "{topic}".

This section scored {score:.0%} on grounding — it contains specific numbers or events
that are NOT in the verified claim registry below.

YOUR TASK:
Rewrite the section so every specific number, score, or named event either:
  a) Comes from the CLAIM REGISTRY — cite it inline as (CLAIM-ID), or
  b) Is replaced with pattern language that makes the same analytical point without the specific detail

PRESERVE the analytical argument, voice, and approximate length. Do NOT add new facts.

CLAIM REGISTRY (only these values are verified):
{registry_block}

SECTION TO REWRITE:
{heading}
{body}

Return ONLY the rewritten section (heading line + body). No commentary.
"""
        try:
            rewritten = _invoke(prompt, temperature=0.2).strip()
            # Strip the heading if LLM duplicated it
            if rewritten.startswith(heading.strip()):
                rewritten = rewritten[len(heading.strip()):].strip()
            sections[idx][1] = rewritten
            logger.info(f"[GROUNDING] Section rewritten OK")
        except Exception as e:
            logger.warning(f"[GROUNDING] Rewrite failed: {e} — keeping original")

    # Reassemble
    rebuilt = preamble
    for heading, body in sections:
        rebuilt += f"\n{heading}\n\n{body}"

    logger.info(f"[GROUNDING] Done | duration={time.time()-start:.1f}s | rewrites={len(sections_to_rewrite)}")
    return rebuilt.strip(), report


def fix_cliches(blog: str, topic: str) -> str:
    found = [c for c in CLICHES if c.lower() in blog.lower()]
    if not found:
        return blog
    cliche_list = "\n".join(f"- {c}" for c in found)
    prompt = f"""
You are a sharp editor. The blog below contains these clichés that must be replaced:

{cliche_list}

For each one, replace it with something specific, fresh, and relevant to the topic "{topic}".
Do not change anything else — only fix the clichés listed above.
Return the full blog with only those replacements made.

Blog:
{blog}
"""
    return _invoke(prompt, temperature=0.5)


@traceable(name="writing_strategy_planner")
def plan_blog(topic: str, audience: str) -> str:
    """
    Elite Writing Strategy Planner.
    Detects topic type, decides writing approach, builds full strategy brief.
    Strategy is embedded directly into the writer — zero extra latency.
    """
    _check_cancelled()
    logger.info(f"[STRATEGY] Starting | topic='{topic[:60]}' | audience='{audience[:40]}'")
    start = time.time()

    result = _invoke(f"""
You are an elite content strategist. Your strategy brief will be used directly by the writer — make it specific, actionable and tight.

Topic: {topic}
Target Audience: {audience}
Audience Note: Write for this specific audience — calibrate depth, tone, and assumed knowledge accordingly.

STEP 0 — READER EMOTIONAL STATE
Arrival Emotion: One word. What emotion does this reader carry in before reading? (e.g. anxiety, skepticism, nostalgia, curiosity)
Departure Feeling: One sentence. What should they FEEL when they close the tab — not what they know, what they feel.
Visualization Depth: SCENE | SENTENCE | STAT
  SCENE = full immersive scene, character + moment + sensory detail (incidents, sports, personal failure)
  SENTENCE = one precise sentence naming a feeling the reader has never seen written (opinion, analytical, life)
  STAT = one number so surprising it creates instant curiosity (science, business, historical)

STEP 1 — TOPIC INTELLIGENCE
Topic Type: (one of) Technical | Sports | Educational | Historical | Analytical | Review | Opinion | Science | Health | Business
Core Question: The single most important question this blog must answer.
Reader Pain Point: What does the reader actually want to understand or solve?
Unique Angle: What do most articles on this topic miss? What makes this one different?

STEP 2 — THE TENSION (this is the most important step)
Answer these two questions with brutal specificity:

Reader's Current Belief: What does this reader already believe or assume about this topic?
(Not what they don't know — what they think they know. The mental model they're walking in with.)

The Challenge: What does this blog need to complicate, correct, or deepen in that belief?
(Not just "add information" — what specifically will make them think differently?)

This tension is the engine of the whole blog. If the blog doesn't create this shift, it's just content.

STEP 3 — WRITING APPROACH
Based on topic type choose the right style:
- Technical: concrete problem first, precise examples, build step by step
- Sports: specific moment first, data supports story, vivid and grounded
- Educational: start simple go deep, analogies, teach progressively
- Historical: set the scene, narrative arc, connect past to present
- Analytical: bold thesis, evidence-backed, challenge assumptions
- Review: honest upfront, fair evaluation, clear verdict
- Opinion: clear stance, acknowledge counterarguments, defend with evidence
- Science/Health: human relevance first, translate complexity, cite evidence
- Business: cost or opportunity first, real numbers, actionable

Writing Style: (name the style)
Why this fits: (one sentence)

STEP 4 — STRUCTURE AND CHAPTER CONTRACT
Opening Hook: (specific — a scene, stat, contradiction or uncomfortable truth for THIS topic)
Core Argument: (the single insight the reader leaves with — must resolve the tension from Step 2)
Conclusion: (how to land it — challenge, reframe, hard truth or memorable final thought)

Sections: list 5-7 sections using this format for EACH one:

SECTION [N]: [Heading]
OWNS: [The one idea this section and ONLY this section develops. One sentence.]
INTRODUCES: [The specific fact, moment, or angle introduced here for the first time.]
FORBIDDEN: [What this section must NOT repeat from previous sections.]
ADVANCES: [How does this section move the central argument forward? What does the reader understand after this section that they didn't before?]

CHAPTER CONTRACT RULES:
- Every section must introduce something new — a new fact, a new angle, a new layer of the argument.
- If a match, event, or fact was used in an earlier section, it cannot be the main focus of a later section.
  It can be referenced briefly only if it is being developed further — not explained again.
- The story moves forward. It never circles back to explain something already explained.
- The reader should feel they are accumulating understanding, not reading the same thing rephrased.

STEP 5 — TONE AND VOICE
Tone: (e.g. analytical-urgent / warm-curious / direct-challenging)
Voice: (e.g. expert coach / investigative journalist / thoughtful teacher)
Avoid: (topic-specific clichés and shallow takes to avoid)

Be specific. Generic strategy produces generic blogs.
""", temperature=0.3)

    logger.info(f"[STRATEGY] Done | duration={time.time()-start:.1f}s")
    return result




@traceable(name="analyze_competitor_gap")
def analyze_competitor_gap(topic: str) -> str:
    """
    Does ONE web search on the topic to see what's already ranking,
    then asks the LLM to identify the angle every competitor takes
    and what unique angle would make this blog stand out.
    Runs in parallel with research — adds zero latency.
    """
    _check_cancelled()
    logger.info(f"[GAP] Starting competitor gap analysis | topic='{topic[:60]}'")
    start = time.time()
    try:
        search_results = web_search(f"{topic} blog article")
    except Exception as e:
        logger.warning(f"[GAP] Web search failed: {e} — skipping gap analysis")
        return ""

    prompt = f"""
You are a content strategist. I searched "{topic}" and found these results:

{search_results}

In 3 to 4 sentences answer:
1. What angle do most of these articles take? (be specific)
2. What important angle, fact, or perspective are they ALL missing?
3. What would make a new blog on this topic genuinely different and more valuable?

Be specific. No fluff. This will be used as a writing brief.
"""
    try:
        result = _invoke(prompt, temperature=0.3)
        logger.info(f"[GAP] Done | duration={time.time()-start:.1f}s")
        return result
    except Exception as e:
        logger.warning(f"[GAP] LLM call failed: {e} — skipping gap analysis")
        return ""


@traceable(name="research")
def research(topic: str, competitor_gap: str = "") -> tuple[str, str, ClaimRegistry]:
    """
    Full 6-phase research engine.
    Returns (dossier_str, raw_facts_str, claim_registry).
    dossier_str    — structured evidence dossier for write_blog
    raw_facts_str  — flat facts list for generate_citations
    claim_registry — ClaimRegistry object for forbidden_pattern_filter
    """
    _check_cancelled()
    logger.info(f"[RESEARCH] Starting research engine | topic='{topic[:60]}'")
    start = time.time()
    dossier, raw_facts, registry = run_research_engine(topic, competitor_gap, _invoke)
    logger.info(f"[RESEARCH] Done | duration={time.time()-start:.1f}s")
    return dossier, raw_facts, registry


@traceable(name="extract_facts")
def extract_facts(topic: str, research_data: str) -> str:
    """
    Backward-compatible shim — raw_facts_str is already built by the research engine.
    This just passes it through so graph.py nodes need minimal changes.
    """
    return research_data


@traceable(name="insight_extractor")
def extract_insights(topic: str, dossier: str, plan: str) -> str:
    """
    Insight Extractor — runs after research, before writing.

    Asks: what are the 3 most interesting, non-obvious things hiding in this research?
    Gives the writer a real IDEA to build the blog around instead of just facts.
    This is the difference between a blog that has a reason to exist vs one that summarises.
    """
    _check_cancelled()
    logger.info(f"[INSIGHTS] Starting | topic='{topic[:60]}'")
    start = time.time()

    dossier_input = dossier[:4_000] if dossier else ""
    plan_input    = plan[:1_500]    if plan    else ""

    prompt = f"""
You are a brilliant editor preparing a writer's brief. A blog will be written about:

TOPIC: "{topic}"

STRATEGY:
{plan_input}

RESEARCH DOSSIER:
{dossier_input}

Your job: find the 3 most interesting, non-obvious insights hiding in this research.
Not the most common facts. Not what everyone already knows.
The stuff that makes a reader stop and think "I never thought about it that way."

For each insight:
- State it as a sharp, specific claim (one sentence)
- Explain WHY it's surprising or counterintuitive (one sentence)
- Note which angle of the topic it unlocks

Format:
INSIGHT 1: [the claim]
WHY IT MATTERS: [why surprising]
UNLOCKS: [angle it opens]

INSIGHT 2: [the claim]
WHY IT MATTERS: [why surprising]
UNLOCKS: [angle it opens]

INSIGHT 3: [the claim]
WHY IT MATTERS: [why surprising]
UNLOCKS: [angle it opens]

CORE THESIS: [one sentence — the single most powerful idea the blog should build toward]

If the dossier is thin, draw on what is widely known but underappreciated about this topic.
Be specific. Be bold. Generic insights are useless.
"""
    result = _invoke(prompt, temperature=0.5)
    logger.info(f"[INSIGHTS] Done | duration={time.time()-start:.1f}s")
    return result


@traceable(name="opening_hook_generator")
def generate_opening_hook(topic: str, insights: str, plan: str, audience: str, narrative: str = "") -> str:
    """
    Opening Hook Generator — one dedicated LLM call, one job.

    Generates 5 candidate opening sentences then picks the sharpest one.
    The opening is the hardest sentence to write and the most important.
    When buried inside a 2000-word prompt the LLM rushes it.
    A dedicated call with a single focus produces a genuinely sharp first sentence.

    Now receives the Narrative Architect output so the hook is the first sentence
    of the actual story — not a standalone attention-grabber that may drift from
    the arc the architect designed.
    """
    _check_cancelled()
    logger.info(f"[HOOK] Generating opening hook | topic='{topic[:60]}'")
    start = time.time()

    plan_input      = plan[:1_000]     if plan      else ""
    insights_input  = insights[:1_000] if insights  else ""

    # Extract the story spine paragraph and emotional arc from the narrative —
    # these are the parts the hook needs to align with, not the full blueprint.
    narrative_input = ""
    if narrative and narrative.strip():
        import re as _re
        # Try to extract PART 2 (story spine) — a single prose paragraph
        spine_match = _re.search(
            r'PART\s+2[^\n]*\n+(.*?)(?=PART\s+3|CORE\s+THESIS|WHAT\s+MUST\s+NOT|$)',
            narrative, _re.IGNORECASE | _re.DOTALL
        )
        arc_match = _re.search(
            r'EMOTIONAL\s+ARC[^\n]*:\s*([^\n]+)', narrative, _re.IGNORECASE
        )
        spine_text = spine_match.group(1).strip()[:600] if spine_match else ""
        arc_text   = arc_match.group(1).strip()         if arc_match   else ""
        if spine_text or arc_text:
            narrative_input = (
                "\nNARRATIVE ALIGNMENT — the hook must be the first beat of this story:\n"
                + (f"Story spine opening: {spine_text}\n" if spine_text else "")
                + (f"Emotional arc: {arc_text}\n"         if arc_text   else "")
            )

    prompt = f"""
You are writing the opening sentence of a blog about: "{topic}"
Target audience: {audience}

CORE INSIGHTS TO DRAW FROM:
{insights_input}

STRATEGY BRIEF:
{plan_input}
{narrative_input}

Generate 5 completely different candidate opening sentences.
Each must be a different TYPE:
1. UNCOMFORTABLE TRUTH — something true that most people sense but nobody says
2. VIVID SCENE — drop the reader into a specific moment, no setup
3. COUNTERINTUITIVE CLAIM — the opposite of what most people assume
4. PROVOCATIVE QUESTION — not a rhetorical cliché, a question that genuinely unsettles
5. SURPRISING STATISTIC OR FACT — one real, specific fact that reframes everything

Rules for every sentence:
- Maximum 25 words
- No throat-clearing ("In today's world...", "Have you ever wondered...", "X is one of the most...")
- No topic name in the first 5 words unless unavoidable
- Must make a smart reader stop scrolling

Then pick the ONE that is sharpest and most likely to hook THIS specific audience.

Format:
TYPE 1 — UNCOMFORTABLE TRUTH: [sentence]
TYPE 2 — VIVID SCENE: [sentence]
TYPE 3 — COUNTERINTUITIVE CLAIM: [sentence]
TYPE 4 — PROVOCATIVE QUESTION: [sentence]
TYPE 5 — SURPRISING FACT: [sentence]

CHOSEN HOOK: [the single best one, copied exactly]
"""
    result = _invoke(prompt, temperature=0.7)

    # Extract just the chosen hook — look for "CHOSEN HOOK:" line
    chosen = ""
    for line in result.split("\n"):
        stripped = line.strip()
        if stripped.upper().startswith("CHOSEN HOOK:"):
            chosen = stripped[len("CHOSEN HOOK:"):].strip().strip('"').strip("'")
            break

    # Fallback: take the last short line that doesn't look like a label
    if not chosen:
        skip_starts = ("type ", "chosen", "reason", "---", "hook ", "**reason", "*reason")
        candidates = [
            l.strip().strip('"').strip("'").strip("*")
            for l in result.split("\n")
            if l.strip()
            and len(l.strip()) > 15
            and len(l.strip()) < 200
            and not any(l.strip().lower().startswith(s) for s in skip_starts)
            and "REASON" not in l.upper()
        ]
        chosen = candidates[-1] if candidates else result[:200]

    logger.info(f"[HOOK] Done | hook='{chosen[:80]}' | duration={time.time()-start:.1f}s")
    return chosen


@traceable(name="ending_generator")
def generate_ending(blog: str, topic: str, insights: str, narrative: str = "") -> str:
    """
    Ending Generator — runs after the draft is written, before citations.
    Upgraded: enforces a NEW insight at the end, not a summary.
    The ending must land on the one thing the whole blog was building toward.
    Returns ONLY the clean final paragraph text — no labels, no metadata.
    """
    _check_cancelled()
    logger.info(f"[ENDING] Generating ending | topic='{topic[:60]}'")
    start = time.time()

    blog_tail      = blog[-1_500:]  if blog     else ""
    insights_input = insights[:800] if insights else ""

    # Extract the payoff beat from narrative if available
    payoff_note = ""
    if narrative and narrative.strip():
        import re as _re
        payoff_match = _re.search(r'(?:payoff|final beat|beat\s*7|last beat)[^\n]*\n([^\n]+)', narrative, _re.IGNORECASE)
        if payoff_match:
            payoff_note = f"\nThe blog was building toward this payoff: {payoff_match.group(1).strip()}\nThe ending MUST land on this or something deeper — not a restatement of it.\n"

    prompt = f"""
You are writing the final section of a blog about: "{topic}"

CORE INSIGHTS THE BLOG BUILT TOWARD:
{insights_input}
{payoff_note}

CURRENT ENDING (what exists — you are REPLACING this):
{blog_tail}

Your job: write an ending that makes the reader feel the whole journey was worth it.

THE ONE RULE THAT MATTERS:
The ending must say something the reader could NOT have predicted from the opening — 
but once they read it, it feels completely inevitable.
Not a summary. Not "in conclusion." Not "this proves that X."
A NEW thought. The thing the blog was quietly building toward.

Write 3 candidate endings (A, B, C). Each takes a different approach:

A — THE REFRAME
Zoom out until the topic means something the reader never expected.
The last sentence should make the reader see the world slightly differently.
The topic is now a lens for something bigger.

B — THE UNCOMFORTABLE TRUTH  
Name the thing the blog has been circling around but never said directly.
The thing that's true, slightly uncomfortable, and impossible to unsay once read.
Land it in one clean, declarative sentence.

C — THE CIRCLE BACK
Return to the opening moment or image — but it means something completely different now.
The reader has changed. Show them how. Don't tell them they've changed — show it
through the new meaning the opening image carries.

RULES FOR ALL THREE:
- 3 to 5 sentences. Never fewer than 3.
- Zero calls to action. Zero motivational fluff.
- Final sentence must be a statement. Never a question.
- Natural rhythm — vary sentence length. Short sentences land harder at the end.
- The ending earns silence. The reader should sit with it for a moment.

After the 3 candidates write exactly: ---CHOSEN---
Then write ONLY the full text of the best ending. No labels. No commentary.
"""
    result = _invoke(prompt, temperature=0.65)

    if "---CHOSEN---" in result:
        final_text = result.split("---CHOSEN---", 1)[1].strip()
    else:
        skip_starts = ("ending a", "ending b", "ending c", "chosen",
                       "final text", "---", "option", "reframe", "hard truth", "circle")
        paragraphs = [p.strip() for p in result.split("\n\n") if p.strip()]
        clean = [p for p in paragraphs
                 if not any(p.lower().startswith(s) for s in skip_starts)]
        final_text = clean[-1] if clean else ""

    skip_prefixes = ("ending a", "ending b", "ending c", "chosen ending",
                     "final text", "---", "option a", "option b", "option c",
                     "reframe:", "hard truth:", "circle back:")
    clean_lines = [
        line for line in final_text.split("\n")
        if not any(line.strip().lower().startswith(p) for p in skip_prefixes)
    ]
    final_text = "\n".join(clean_lines).strip()

    logger.info(f"[ENDING] Done | chars={len(final_text)} | duration={time.time()-start:.1f}s")
    return final_text


@traceable(name="narrative_architect")
def narrative_architect(topic: str, audience: str, insights: str, plan: str, length: str = "medium") -> str:
    """
    Narrative Architect — the backbone of elite writing.

    Does NOT hardcode shapes (sports→moment-first, etc.).
    Instead: gives the LLM the topic, audience, insights, and asks it to reason
    from first principles about the optimal story shape for THIS specific blog.

    Output: a beat-by-beat story outline where every beat has:
    - A job (what it must make the reader feel or understand)
    - The narrative technique to use
    - A transition hook that pulls the reader to the next beat

    The writer follows this outline as a contract, not suggestions.
    """
    _check_cancelled()
    logger.info(f"[ARCHITECT] Building narrative outline | topic='{topic[:60]}'")
    start = time.time()

    insights_input = insights[:1_500] if insights else ""
    plan_input     = plan[:1_000]     if plan     else ""

    beat_count = {"short": 4, "medium": 6, "long": 8}.get(length, 6)

    prompt = f"""
You are a narrative architect. Your job is to feel the story before the writer writes it.

TOPIC: "{topic}"
AUDIENCE: {audience}
LENGTH: {length} ({beat_count} beats)

CORE INSIGHTS:
{insights_input}

STRATEGY BRIEF:
{plan_input}

═══════════════════════════════════════
PART 1 — FEEL THE STORY FIRST
═══════════════════════════════════════

Before anything structural, answer these three questions in plain sentences.
Not bullet points. Not labels. Write like a person thinking.

WHAT THE READER WALKS IN BELIEVING:
Write one paragraph. What does this reader already think about this topic?
Not what they don't know — what they think they know. Their existing mental model.
Be specific to this exact reader and this exact topic.

WHAT THIS BLOG DOES TO THAT BELIEF:
Write one paragraph. This blog doesn't just add information to their existing belief —
it complicates it, challenges it, or deepens it in a way they didn't expect.
Name the specific shift. Not "they'll learn more." What will they think differently?

THE EMOTIONAL ARC IN ONE SENTENCE:
How does the reader feel at the START vs the END?
Not what they know — what they feel.
Example: "They arrive skeptical and leave with a quiet respect they can't entirely explain."
Example: "They arrive certain and leave with a productive doubt."
Write yours in one sentence.

═══════════════════════════════════════
PART 2 — THE STORY SPINE
═══════════════════════════════════════

Now write the story as a single continuous paragraph — {beat_count} sentences,
one sentence per beat. This is the story told in compressed form.

Do NOT use labels like "Beat 1" or "Section 2".
Do NOT use bullet points or structure.
Write it the way you'd tell a friend what this blog is about — as a flowing narrative.

Each sentence should:
- Do something specific to the reader (create tension, deliver a surprise, reframe what came before)
- Pull naturally into the next sentence
- Build toward the final sentence, which lands the whole thing

The first sentence must grab. The last sentence must land.
Everything in between must pull.

STORY ANCHOR (one line, after the spine paragraph):
Name the single concrete scene or character the writer opens with and must return to in at least 3 chapters.
Not a theme — a specific moment. A person in a place with a problem.
Format: STORY ANCHOR: [person/character] + [specific situation] + [what's at stake]
Example: STORY ANCHOR: A fintech engineer, 2am, watching a compliance bot answer questions using retired policy rules.
The writer uses this as the thread the whole blog hangs on.

═══════════════════════════════════════
PART 3 — THE CONTRACT
═══════════════════════════════════════

After the story spine, write three things:

CORE THESIS (one sentence):
The single idea the entire blog is building toward. Not a topic description.
The specific thought the reader should be thinking as they close the tab.

WHAT MUST NOT HAPPEN:
Name the one failure mode that would kill this blog.
e.g. "The writer explains instead of shows and the reader feels lectured."
e.g. "The sections feel like separate arguments instead of one accumulating proof."
e.g. "The ending summarises instead of landing something new."
One sentence. The writer should read this and feel the pressure of it.

TONE IN ONE LINE:
Not a list of adjectives. One sentence that captures exactly how this should sound.
e.g. "Write like someone who has watched every game and is finally allowed to say what they actually think."
e.g. "Write like a detective who already knows the answer but is walking the reader through the evidence."

Be specific. This spine is what the writer feels while writing — not a checklist to execute.
"""
    result = _invoke(prompt, temperature=0.4)
    logger.info(f"[ARCHITECT] Done | chars={len(result)} | duration={time.time()-start:.1f}s")
    return result


@traceable(name="unified_writer")
def write_blog(topic: str, audience: str, plan: str, research_data: str, memory: str, length: str = "medium", gap: str = "", facts: str | None = None, hook: str = "", insights: str = "", registry: "ClaimRegistry | None" = None, narrative: str = "") -> str:
    """
    Unified Writer — Paul Graham + Tim Urban + George Orwell combined.
    Adaptive by topic type — reading topic type from Strategy Planner output.
    """
    _check_cancelled()
    logger.info(f"[WRITER] Starting | topic='{topic[:60]}' | length={length}")
    start = time.time()
    verified_facts = facts if facts is not None else research_data
    length_instruction = LENGTH_GUIDE.get(length, LENGTH_GUIDE["medium"])
    gap_section = f"\nCompetitor Gap — what existing articles miss (address this angle):\n{gap}\n" if gap else ""

    # Cap research inputs to avoid 413 token errors
    # dossier at 5 000 chars, facts at 2 000 chars
    verified_facts_input = verified_facts[:5_000] if verified_facts else ""
    plan_input           = plan[:3_000] if plan else ""

    # Extract topic type from strategy plan for adaptive mechanic weighting
    import re as _re
    topic_type_match = _re.search(r'topic type[:\s]+([A-Za-z]+)', plan, _re.IGNORECASE)
    topic_type = topic_type_match.group(1).strip().lower() if topic_type_match else "general"

    # Adaptive mechanic weights per topic type
    MECHANIC_WEIGHTS = {
        "sports":      ("Tim Urban — storytelling and vivid moments first", "Orwell — precision and concrete details", "Graham — one surprising insight per section"),
        "technical":   ("Graham — reasoning chain and why it works", "Orwell — clarity, concrete, active voice", "Urban — one strong analogy per concept"),
        "educational": ("Urban — curiosity, zoom out, progressive build", "Orwell — simplest word, concrete examples", "Graham — one insight the textbook misses"),
        "historical":  ("Urban — narrative arc, scene-setting, scale", "Graham — connect past to present with an original insight", "Orwell — precise, no filler"),
        "analytical":  ("Graham — contrarian insight, show the reasoning chain", "Orwell — concrete evidence, strong verbs", "Urban — one analogy to make the argument land"),
        "opinion":     ("Graham — uncomfortable truth, defend with reasoning", "Orwell — precise, no wasted words, active voice", "Urban — relatable frame and villain"),
        "science":     ("Urban — human relevance first, bridge analogies", "Orwell — translate complexity into concrete clarity", "Graham — why does this actually work?"),
        "health":      ("Urban — human relevance first, bridge analogies", "Orwell — translate complexity into concrete clarity", "Graham — why does this actually work?"),
        "business":    ("Graham — original insight, cost or opportunity framing", "Orwell — real numbers, active voice, no jargon", "Urban — relatable villain, reader frame"),
        "review":      ("Orwell — honest, concrete, clear verdict upfront", "Graham — reasoning behind the verdict", "Urban — reader's actual situation frame"),
    }

    weights = MECHANIC_WEIGHTS.get(topic_type, (
        "Graham — original insight and reasoning",
        "Urban — curiosity and one strong analogy",
        "Orwell — clarity and concrete language"
    ))
    primary, secondary, light = weights

    # Visualization rule — domain-aware, one before/after pair, token-efficient.
    # The rule is the same for every domain: don't describe, put the reader inside.
    # The example changes per domain so it lands concretely for this topic type.
    _VIZ_EXAMPLES = {
        "technical":   ("OBSERVED (banned): 'The deploy failed and latency spiked.'\nVISUALIZED: 'It's 2am. You pushed four hours ago. Your phone won't stop. The bot is answering with rules retired in March.'"),
        "sports":      ("OBSERVED (banned): 'He faced a difficult final over under pressure.'\nVISUALIZED: 'Six needed. Last ball. His hands aren't shaking — that's the thing nobody in the stands understands.'"),
        "business":    ("OBSERVED (banned): 'The startup made a pricing mistake that hurt growth.'\nVISUALIZED: 'The spreadsheet looked right. The first churn email arrived eleven days after launch.'"),
        "health":      ("OBSERVED (banned): 'Many people struggle with sleep deprivation.'\nVISUALIZED: 'You know you need eight hours. You also know exactly how 11pm becomes 1am.'"),
        "analytical":  ("OBSERVED (banned): 'The data revealed an unexpected pattern.'\nVISUALIZED: 'The number sat in the report for six months. Nobody asked why it was wrong.'"),
        "opinion":     ("OBSERVED (banned): 'Many people believe remote work is productive.'\nVISUALIZED: 'Your best thinking happened at 6am before anyone else logged on. That is the data.'"),
        "science":     ("OBSERVED (banned): 'Studies show exercise improves cognition.'\nVISUALIZED: 'Twenty minutes in, the problem you have been staring at for two hours solves itself.'"),
        "historical":  ("OBSERVED (banned): 'The decision had significant consequences.'\nVISUALIZED: 'The telegram arrived at noon. By evening three countries had changed their position.'"),
        "educational": ("OBSERVED (banned): 'Compound interest grows wealth over time.'\nVISUALIZED: 'The account you opened at 22 and forgot about — that is the one doing the most work.'"),
    }
    viz_example = _VIZ_EXAMPLES.get(topic_type, (
        "OBSERVED (banned): 'The situation was difficult and had real consequences.'\n"
        "VISUALIZED: 'Name the person. Name the moment. Name what was at stake. Put the reader there.'"
    ))
    visualization_rule = f"""Don't describe what happens. Put the reader inside it.
The reader is not watching a scene — they are living it.

{viz_example}

Rule: Every scene-setting sentence must pass this test — is the reader IN this, or watching it?
If watching: rewrite until they're in."""

    # Voice profile per topic type.
    # These are not job descriptions — they are writing postures.
    # Each one tells the LLM HOW to hold the topic, not WHO to pretend to be.
    # The difference: "you are a fan" produces performed enthusiasm.
    # "You find one thing about this genuinely unfair" produces real voice.
    VOICE_PROFILES = {
        "sports": (
            "You find something about this topic genuinely unfair or underappreciated — "
            "and you've been wanting to say it. You don't describe what happened. "
            "You argue about what it means. You write like someone who has watched "
            "every game, read every stat, and is finally allowed to say what they actually think. "
            "Short sentences when making a point. Longer ones when building the case. "
            "Never neutral. Always specific."
        ),
        "technical": (
            "You write like a practitioner who has hit this problem in production, "
            "not a textbook author explaining theory. You skip what's obvious to anyone "
            "who has been in the field. You go straight to the thing most tutorials miss. "
            "Concrete over abstract, always. The moment you catch yourself using a vague word, "
            "you replace it with the specific thing you actually mean."
        ),
        "educational": (
            "You write like someone who just had a genuine insight about this topic "
            "and can't stop thinking about it. Not excitement — curiosity. "
            "You let the reader discover things slightly before you name them. "
            "You ask the question the reader is already forming, then answer it better than they expected. "
            "You never explain what someone already knows. You only explain the gap."
        ),
        "historical": (
            "You write like someone who finds the past genuinely strange — "
            "and can't understand why everyone treats it as settled. "
            "You put the reader in the room. You care about why things happened, "
            "not just what happened. Every historical fact connects to something "
            "the reader can feel today, or it doesn't belong in the blog."
        ),
        "analytical": (
            "You write like a detective who already knows the answer "
            "and is walking the reader through the evidence. "
            "You say the uncomfortable thing plainly. You show your reasoning — "
            "not just your conclusions. When you make a claim, you immediately "
            "ask yourself 'what would someone say to push back on this?' "
            "and you answer it before they can."
        ),
        "opinion": (
            "You have a position and you hold it. You don't hedge. "
            "You name the opposing view clearly — then you dismantle it with evidence, "
            "not dismissal. You write like someone who has thought about this longer "
            "than everyone else in the room and is tired of the shallow takes. "
            "Direct. Specific. Never mean, but never soft either."
        ),
        "science": (
            "You write like a translator — from research language to human language. "
            "You find the moment where the science becomes surprising, "
            "then you start there. You never use jargon without immediately showing "
            "what it looks like in real life. You are precise because precision is "
            "interesting, not because it sounds authoritative."
        ),
        "health": (
            "You write like someone who has read the studies and is telling a friend "
            "what they actually say — not what the headlines say. "
            "You respect the reader's intelligence. You say 'we don't know yet' "
            "when that's true. You say 'this is well-established' when that's true. "
            "No fear. No hype. Just what the evidence actually means for someone's life."
        ),
        "business": (
            "You write like someone who has sat in the room where the decision was made. "
            "You cut through the framework language. You talk about money, "
            "incentives, and tradeoffs the way operators talk in private — "
            "not the way consultants talk on stage. "
            "Every abstract claim gets a concrete number or a named example, or it gets cut."
        ),
        "review": (
            "You have a verdict and you give it in the first paragraph. "
            "Everything after is evidence. You write like a trusted friend "
            "who has actually used this thing — not a press release, "
            "not a hedge. You name what's good specifically. "
            "You name what's bad specifically. You tell the reader "
            "exactly who this is for and who should walk away."
        ),
        "concept_explainer": (
            "You find this idea genuinely fascinating — not because it's impressive, "
            "but because it changes how you see something you thought you understood. "
            "You don't lecture. You show the reader something, then stand back "
            "and let them feel the strangeness of it. "
            "You let the idea do the work. You just clear the path."
        ),
        "person_biography": (
            "You write like someone who has figured out the one thing "
            "about this person that everyone else keeps missing. "
            "Not their achievements — the specific way they think, "
            "or decide, or fail, or recover. That's the story. "
            "Everything else is context for it."
        ),
        "historical_event": (
            "You put the reader in the room. Not the textbook summary — "
            "the actual moment, with the actual stakes. "
            "You make the reader feel what it meant to be there, "
            "not just what happened next. And then you connect it "
            "to something they can feel today — or the history is just trivia."
        ),
    }

    voice = VOICE_PROFILES.get(
        topic_type,
        (
            "You have thought about this topic more carefully than most people, "
            "and you have something specific to say about it. "
            "Not a summary. Not a survey. One clear point of view, "
            "built carefully, sentence by sentence. "
            "You write like a person — not a content generator."
        )
    )

    # Opening hook section — if a pre-generated hook exists, the writer leads with it
    hook_section = ""
    if hook and hook.strip():
        hook_section = f"""
OPENING HOOK — START WITH THIS EXACT SENTENCE:
{hook.strip()}

Do not change it. Do not add "Introduction" before it. The blog starts with this sentence.
"""

    # Insights section
    insights_section = ""
    if insights and insights.strip():
        insights_section = f"""
CORE INSIGHTS TO BUILD THE BLOG AROUND:
{insights[:2_000]}

One of these should be the spine of the whole blog. Pick the one that fits best and develop it deeply.
Don't just mention all three and move on — choose one as the central argument and let the others support it.
"""

    # Narrative Architect spine — the felt story the writer carries while writing.
    # We pass PART 2 (story spine) and PART 3 (contract) in full — these are the
    # sections the writer needs most. PART 1 (reader belief analysis) is included
    # only if budget allows. Total cap raised to 5,000 chars so the full blueprint
    # reaches the writer even for long blogs.
    narrative_section = ""
    if narrative and narrative.strip():
        import re as _re2

        # Try to extract PART 2 and PART 3 specifically — highest value for the writer
        part2_match = _re2.search(
            r'(PART\s+2[^\n]*\n+.*?)(?=PART\s+3|$)',
            narrative, _re2.IGNORECASE | _re2.DOTALL
        )
        part3_match = _re2.search(
            r'(PART\s+3[^\n]*\n+.*?)$',
            narrative, _re2.IGNORECASE | _re2.DOTALL
        )
        part2_text = part2_match.group(1).strip() if part2_match else ""
        part3_text = part3_match.group(1).strip() if part3_match else ""

        if part2_text or part3_text:
            # Compose from extracted sections with generous individual caps
            narrative_body = ""
            if part2_text:
                narrative_body += part2_text[:2_500] + "\n\n"
            if part3_text:
                narrative_body += part3_text[:2_500]
            narrative_body = narrative_body.strip()
        else:
            # Fallback: no PART markers found — send full narrative up to 5,000 chars
            narrative_body = narrative[:5_000]

        # Extract STORY ANCHOR — the concrete thread the writer must return to
        anchor_match = _re2.search(r'STORY ANCHOR:\s*([^\n]+)', narrative, _re2.IGNORECASE)
        anchor_line = f"\nSTORY ANCHOR — return to this in every chapter: {anchor_match.group(1).strip()}\n" if anchor_match else ""

        narrative_section = f"""
═══════════════════════════════════════
THE STORY YOU ARE TELLING
═══════════════════════════════════════
{anchor_line}
{narrative_body}

Every paragraph either moves this story forward or it doesn't belong.
Sections are moments in a continuous journey — not containers for information.
If the reader could skip a section and not feel the loss — rewrite it.
"""

    # Audience-aware depth instruction — tells the writer what to assume about
    # the reader's existing knowledge so it doesn't over-explain or under-explain.
    audience_lower = audience.lower()
    if any(w in audience_lower for w in ["fan", "enthusiast", "follower", "player", "expert", "professional", "developer", "engineer"]):
        audience_depth = (
            f"Audience depth: KNOWLEDGEABLE — {audience} already understand the basics. "
            "Skip definitions. Use insider language. Go one level deeper than a general explainer."
        )
    elif any(w in audience_lower for w in ["beginner", "new", "first", "learning", "curious", "general"]):
        audience_depth = (
            f"Audience depth: GENERAL — {audience} may be new to parts of this topic. "
            "Explain jargon the first time you use it. Use analogies. Don't assume prior knowledge."
        )
    else:
        audience_depth = (
            f"Audience depth: MIXED — {audience}. Explain the non-obvious, skip the obvious. "
            "Write like a smart friend who knows this topic well, not a textbook."
        )

    # ── Claim Registry block for writer ──────────────────────────────────────
    if registry is not None and registry.entries:
        registry_section = registry.to_writer_block()
    else:
        registry_section = "CLAIM REGISTRY: No verified claims available — write without specific numbers, scores, or attributed quotes."

    prompt = f"""
{voice}

Your job: write a blog that a real person actually wants to read. Not a report. Not a Wikipedia summary. Not a coaching manual. A blog — something with a voice, a point of view, and a reason to exist.

Topic: {topic}
Target Audience: {audience}
{audience_depth}
Length: {length_instruction}

WRITING STRATEGY:
{plan_input}
{gap_section}
{hook_section}
{insights_section}
{narrative_section}

RESEARCH DOSSIER — FACTS YOU CAN USE:
{verified_facts_input}

PAST CONTEXT:
{memory}

═══════════════════════════════════════
WHAT A BLOG IS
═══════════════════════════════════════

A blog IS:
- A writer with a specific point of view talking to a specific person
- ONE central argument or insight the whole piece builds toward
- Something that makes the reader think "I never thought about it that way"
- Sections that each reveal something NEW — not just more of the same
- Rhythm — some paragraphs punch, some flow, some ask a question and sit with it
- An ending that lands. The reader feels something.

A blog is NOT:
- Identical sections each ending with the same structural element
- A list of facts with headers
- Five paragraphs that all do the same thing
- A report that describes what experts say without having an opinion

═══════════════════════════════════════
VISUALIZATION RULE
═══════════════════════════════════════
{visualization_rule}

═══════════════════════════════════════
ADAPTIVE MECHANICS: {topic_type.upper()}
═══════════════════════════════════════

Primary: {primary}
Secondary: {secondary}
Light use: {light}

PAUL GRAHAM MECHANICS (use these concretely):
- Start with the most uncomfortable true thing about this topic
- Show WHY the obvious answer is wrong before giving the real answer
- Find the non-obvious insight — the thing that makes the reader think differently
- Defend your position. Don't hedge. Say what you actually think.
- Explain the implication: "which means..." — follow the idea to its conclusion

TIM URBAN MECHANICS (use these concretely):
- Find the reader's existing mental model — then complicate it
- Zoom out until the scale becomes surprising. Make the familiar feel strange.
- One bridge analogy per blog — physical, visual, specific. Not "it's like a machine."
- End sections with the question the reader is now asking. Let them feel the pull.
- "You" frame — the reader is in the story, not watching from outside

ORWELL MECHANICS (use these concretely):
- Every abstract word replaced by a concrete one. No "impactful", "leverage", "synergy".
- Active voice always. "The ball moved" not "the ball was moved."
- Strong verbs. Not "he was running" — "he sprinted." Not "she felt nervous" — "her hands stopped."
- Rhythm: Never three sentences of the same length in a row. Vary deliberately.
  After a long sentence — a short one. Sometimes one word. Sometimes a fragment.
- Every sentence earns its place. If it repeats something already said — cut it.

BANNED WORDS: utilize, leverage, impactful, holistic, synergy, paradigm,
"it's important to note", "in today's world", "needless to say",
"at the end of the day", "game changer", "deep dive", "journey", "testament"

BANNED ABSTRACT PHRASES: "the crucible of", "transcends mere", "forged in the fire",
"the anatomy of", "statistical fingerprint", "the weight of a single",
"lighthouse in a storm", "legacy reshaped", "redefines greatness"
When you catch yourself using any phrase where a physical noun stands in for an abstract idea — replace it with the concrete thing you actually mean.

WRITING CHECKS (run after each paragraph):
- Rhythm: no three consecutive sentences at the same length. Vary deliberately. Short sentences land.
- No throat-clearing openers: "This shows...", "It is worth noting...", "As we can see..." — delete them.
- No significance announcements: "This is crucial", "This reveals", "This demonstrates" — delete them. The paragraph before should already make the reader feel it.
- One stake per section: find the thing about this point that is surprising or counterintuitive. Write from that place.

═══════════════════════════════════════
STRUCTURE
═══════════════════════════════════════

- Short paragraphs — 2 to 4 lines max. Blank line between paragraphs.
- Section headings: ### before each heading
- No bullet points or numbered lists in the main body
- No emojis, no hashtags
- Never open a section by announcing what it covers. Start mid-argument — a fact, scene, claim, or contradiction.

═══════════════════════════════════════
FORWARD-ONLY — NO REPETITION
═══════════════════════════════════════

Every section introduces something new. Once a match or idea is used — it's done.
ONE MATCH = ONE SECTION. After that, one-clause reference only. Never re-explain.
Before writing each section ask: "What does this introduce that nothing before it has?"
If the answer is nothing new — deepen the existing idea or move on entirely.

═══════════════════════════════════════
FACT RULES — CLAIM-FIRST WRITING
═══════════════════════════════════════

{registry_section}

BLACKLISTED — only write these if a CLAIM-ID exists above:
  ✗ Specific scores, results, over outcomes, direct quotes, named event + scoreline
If no CLAIM-ID — describe the pattern, not the specific instance.
If a stat IS in the registry — use it and cite (CLAIM-ID) inline.

═══════════════════════════════════════
SCENE-FIRST RULE
═══════════════════════════════════════

Before explaining any important moment — show it.
Answer these before explaining significance: What was the situation? What made it hard?
What was the person trying to do? How did it feel?
Don't explain it. Make the reader feel it first. Then explain it.

═══════════════════════════════════════
THE ONE TEST
═══════════════════════════════════════

Before writing each section: would a passionate reader learn something, feel something, or see something differently?
If no — rewrite until yes.

Write the full blog now. {"Start with the hook above." if hook else "Start with a scene, tension, uncomfortable truth, or question. Not the topic name. Not a definition. Not 'Introduction'."}
"""
    result = _invoke(prompt, temperature=0.65)
    logger.info(f"[WRITER] Done | topic_type={topic_type} | duration={time.time()-start:.1f}s | words={len(result.split())}")
    return result


@traceable(name="audit_facts")
def audit_facts(blog: str, dossier: str, topic: str) -> str:
    """
    Surgical fact audit — Python-first, LLM-second.

    Step 1 (Python): Extract every specific number from the blog and check
    if that number appears anywhere in the dossier. If ALL numbers are
    verified, skip the LLM call entirely (saves ~45s per blog).

    Step 2 (LLM, only if unverified numbers exist): Send ONLY the sentences
    containing unverified numbers, ask LLM to rephrase just those sentences.
    Never touch the rest of the blog.

    Design principle: narrative, analysis, and opinion are never touched.
    Only falsifiable specific numbers can be acted on.
    """
    import re as _re
    _check_cancelled()
    logger.info(f"[AUDIT] Starting fact audit | topic='{topic[:60]}'")
    start = time.time()

    # ── Step 1: Python number scan ────────────────────────────────────────────
    # Find all specific numbers in the blog (integers and decimals)
    number_pattern = _re.compile(r'\b\d[\d,]*(?:\.\d+)?(?:\s*%|\s*(?:runs|wickets|matches|years|km|mph|kmh|km/h|lbs|kg|crore|million|billion|goals|titles|trophies))?\b')
    all_numbers = number_pattern.findall(blog)
    # Normalize: strip commas, lowercase
    all_numbers_clean = [n.replace(",", "").strip() for n in all_numbers]

    if not all_numbers_clean:
        logger.info(f"[AUDIT] No specific numbers found — skipping | duration={time.time()-start:.1f}s")
        return blog

    dossier_lower = dossier.lower()
    unverified = []
    for num in all_numbers_clean:
        num_lo = num.lower()
        # Short numbers (1-2 digits, e.g. "12", "85") are common and can match
        # unrelated dossier text. Require them to appear with adjacent context
        # (the word before or after in the blog) to prevent false passes.
        if len(num_lo.replace(".", "").replace("%", "")) <= 2:
            # Find the sentence containing this number in the blog
            sentence_containing = next(
                (s for s in _re.split(r'(?<=[.!?])\s+', blog)
                 if num_lo in s.replace(",", "").lower()),
                ""
            )
            # Extract 1-2 surrounding words for context
            context_match = _re.search(
                r'(\w+\s+)?' + _re.escape(num_lo) + r'(\s+\w+)?',
                sentence_containing.lower()
            )
            if context_match:
                context_str = context_match.group(0).strip()
                if context_str not in dossier_lower:
                    unverified.append(num)
            else:
                # Can't build context — treat as unverified
                unverified.append(num)
        else:
            # Longer numbers: direct match is specific enough
            if num_lo not in dossier_lower:
                unverified.append(num)

    # Deduplicate
    unverified = list(dict.fromkeys(unverified))

    if not unverified:
        logger.info(
            f"[AUDIT] All {len(all_numbers_clean)} numbers verified against dossier — "
            f"no LLM call needed | duration={time.time()-start:.1f}s"
        )
        return blog

    logger.info(f"[AUDIT] Found {len(unverified)} unverified numbers: {unverified[:10]}")

    # ── Step 2: LLM surgical fix — only unverified numbers ───────────────────
    # Extract ONLY the sentences containing those unverified numbers
    sentences_to_fix = []
    for sentence in _re.split(r'(?<=[.!?])\s+', blog):
        for num in unverified:
            if num in sentence.replace(",", ""):
                sentences_to_fix.append(sentence.strip())
                break

    if not sentences_to_fix:
        return blog

    # Deduplicate sentences
    sentences_to_fix = list(dict.fromkeys(sentences_to_fix))
    sentences_block = "\n".join(f"- {s}" for s in sentences_to_fix[:15])  # cap at 15

    prompt = f"""
You are a surgical editor. Remove ONLY the unverified numbers from these specific sentences.

Topic: {topic}
Unverified numbers to remove: {unverified}

Sentences to fix:
{sentences_block}

For each sentence:
- Remove the unverified number
- Keep the rest of the sentence intact and natural
- If removing the number makes the sentence grammatically broken or meaningless
  (e.g. "the innings lasted balls" or "stride never exceeded seconds"),
  rewrite the whole sentence WITHOUT the number to preserve meaning
- Do NOT add any new facts
- Return: original sentence → fixed sentence (one per line)

Format:
ORIGINAL: [original sentence]
FIXED: [fixed sentence]
"""
    try:
        raw = _invoke(prompt, temperature=0.0)
    except Exception as e:
        logger.warning(f"[AUDIT] LLM fix call failed: {e} — keeping original blog")
        return blog

    # ── Apply number fixes back to the blog ───────────────────────────────────
    corrected_blog = blog
    for block in raw.strip().split("\n\n"):
        lines = block.strip().split("\n")
        orig_line = next((l[len("ORIGINAL:"):].strip() for l in lines if l.startswith("ORIGINAL:")), None)
        fixed_line = next((l[len("FIXED:"):].strip() for l in lines if l.startswith("FIXED:")), None)
        if orig_line and fixed_line and orig_line != fixed_line:
            corrected_blog = corrected_blog.replace(orig_line, fixed_line, 1)

    logger.info(f"[AUDIT] Number fixes applied | duration={time.time()-start:.1f}s")

    # ── Step 3: Named claims and attributed quotes audit ─────────────────────
    # Extract sentences containing:
    #   a) Quoted speech attributed to a real person  ("X said...", "X described...")
    #   b) Named specific events ("In the 2024 IPL semi-final...", "During the World Cup final against...")
    # Cross-check each against the dossier. If not found — fix it.
    corrected_blog = _audit_named_claims(corrected_blog, dossier, topic)

    logger.info(f"[AUDIT] Done | total_duration={time.time()-start:.1f}s")
    return corrected_blog


def _audit_named_claims(blog: str, dossier: str, topic: str) -> str:
    """
    Step 3 of audit_facts.

    Detects two dangerous hallucination types:
    1. Attributed quotes — sentences where a named person is quoted directly
       e.g. 'Warner described it as "standing on a tightrope"'
    2. Hyper-specific named events — sentences anchoring a claim to a specific
       named match, year, and outcome that may have been invented
       e.g. 'In the 2024 IPL semi-final, Bumrah bowled a 19-run over'

    For each suspicious sentence, checks if the core claim exists in the dossier.
    If not found — sends only those sentences to the LLM to generalize.
    Leaves everything else untouched.
    """
    import re as _re

    if not dossier or not dossier.strip():
        logger.info("[AUDIT-CLAIMS] No dossier — skipping named claims audit")
        return blog

    dossier_lower = dossier.lower()

    # ── Pattern 1: Attributed quotes ─────────────────────────────────────────
    # Matches: Name + said/described/called/noted/wrote/told/explained + quote
    quote_attribution = _re.compile(
        r'(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
        r'(?:said|described|called|noted|wrote|told|explained|stated|admitted|revealed|claimed|added|replied)'
        r'[^.!?]*["\u201c\u2018][^"\u201d\u2019]{5,}["\u201d\u2019]',
        _re.IGNORECASE
    )

    # ── Pattern 2: Hyper-specific named events ────────────────────────────────
    # Matches: "In the 2023 World Cup final", "During the 2024 IPL semi-final"
    named_event = _re.compile(
        r'(?:In|During|At)\s+the\s+(?:\d{4}\s+)?'
        r'(?:IPL|World Cup|Champions Trophy|T20I|ODI|Test|Grand Prix|'
        r'Premier League|NBA|Super Bowl|Ashes|series|final|semi-final|quarter-final)'
        r'[^.!?]*',
        _re.IGNORECASE
    )

    # Split blog into sentences for targeted scanning
    sentences = _re.split(r'(?<=[.!?])\s+', blog)
    suspicious: list[str] = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        is_quoted = bool(quote_attribution.search(sentence))
        is_named_event = bool(named_event.search(sentence))

        if not (is_quoted or is_named_event):
            continue

        # Check if core nouns from the sentence appear in dossier
        # Extract capitalized words (likely proper nouns / names)
        proper_nouns = _re.findall(r'\b[A-Z][a-z]{2,}\b', sentence)
        # For events, also check that any numbers in the sentence appear in the dossier
        # (e.g. "2024 IPL semi-final" — the year "2024" must be in the dossier too)
        event_numbers = _re.findall(r'\b\d{4}\b', sentence)  # years like 2024, 2023
        numbers_verified = all(n in dossier_lower for n in event_numbers) if event_numbers else True

        # Check if at least 2 proper nouns from this sentence appear in dossier
        matched = sum(1 for noun in proper_nouns if noun.lower() in dossier_lower)

        if matched < 2 or not numbers_verified:
            # Likely invented — flag for fixing
            suspicious.append(sentence)

    if not suspicious:
        logger.info("[AUDIT-CLAIMS] No suspicious named claims found")
        return blog

    # Deduplicate
    suspicious = list(dict.fromkeys(suspicious))
    logger.info(f"[AUDIT-CLAIMS] Found {len(suspicious)} suspicious named claims/quotes")

    sentences_block = "\n".join(f"- {s}" for s in suspicious[:10])  # cap at 10

    prompt = f"""
You are a fact-safety editor. The sentences below contain attributed quotes or specific named events that could not be verified against the research dossier.

Topic: {topic}

Your job: rewrite each sentence to remove the unverifiable specific detail while keeping the underlying point.

Rules:
- Remove direct quotes from named people unless the quote appears verbatim in the dossier
- Replace hyper-specific named events ("In the 2024 IPL semi-final...") with pattern-based language ("In high-pressure death overs..." or "Across multiple high-stakes matches...")
- Keep the analytical point — just remove the invented specificity
- Do NOT add new facts
- Do NOT change sentences that are clearly fine (opinions, general descriptions)

Sentences to fix:
{sentences_block}

Format — one block per sentence:
ORIGINAL: [original sentence]
FIXED: [fixed sentence]
"""

    try:
        raw = _invoke(prompt, temperature=0.0)
        logger.info("[AUDIT-CLAIMS] LLM fix call succeeded")
    except Exception as e:
        logger.warning(f"[AUDIT-CLAIMS] LLM fix call failed: {e} — keeping original")
        return blog

    # Apply fixes
    fixed_blog = blog
    for block in raw.strip().split("\n\n"):
        lines = block.strip().split("\n")
        orig = next((l[len("ORIGINAL:"):].strip() for l in lines if l.startswith("ORIGINAL:")), None)
        fixed = next((l[len("FIXED:"):].strip() for l in lines if l.startswith("FIXED:")), None)
        if orig and fixed and orig != fixed:
            fixed_blog = fixed_blog.replace(orig, fixed, 1)
            logger.info(f"[AUDIT-CLAIMS] Fixed: '{orig[:60]}...' → '{fixed[:60]}...'")

    return fixed_blog


@traceable(name="editorial_critic")
def critique_and_rewrite(blog: str, topic: str, audience: str, narrative: str = "") -> str:
    """
    Story Critic — upgraded from quality checker to narrative diagnostician.
    Diagnoses story gaps, not just surface quality.
    Asks where the reader's journey broke down, not just whether the blog is good.
    """
    _check_cancelled()
    logger.info(f"[CRITIC] Starting | topic='{topic[:60]}'")
    start = time.time()

    # Include the narrative blueprint so the critic can check if beats were executed
    narrative_ref = ""
    if narrative and narrative.strip():
        narrative_ref = f"""
NARRATIVE BLUEPRINT (what the blog was supposed to do):
{narrative[:1_500]}

Check whether the writer actually executed each beat's JOB.
"""

    # Pre-compute blog slice — avoids backslash-in-f-string on Python 3.10
    _ELLIPSIS = "\n...\n"
    _blog_for_critic = blog if len(blog) < 8_000 else blog[:5_500] + _ELLIPSIS + blog[-1_500:]

    prompt = f"""
You are a story critic — not a proofreader. You diagnose where the reader's journey broke down.

Topic: {topic}
Audience: {audience}
{narrative_ref}

Read the blog below and answer these 6 diagnostic questions. Be specific and brutally honest.

─────────────────────────────────────────
Q1 — DID THE OPENING CREATE A CONTRACT?
─────────────────────────────────────────
The opening should make a promise — a question, tension, or idea the reader needs resolved.
Did it? Or did it just introduce the topic?
Answer: YES / NO
If NO — what's missing and write a sharper opening (1-2 sentences max):

─────────────────────────────────────────
Q2 — IS THERE A NARRATIVE SPINE?
─────────────────────────────────────────
Can you trace one continuous idea or tension from Beat 1 to the final section?
Or does the blog feel like separate sections stacked on top of each other?
Answer: SPINE / STACKED
If STACKED — name the section where the thread breaks and what it should connect to:

─────────────────────────────────────────
Q3 — WHERE DID THE WRITER PLAY IT SAFE?
─────────────────────────────────────────
Find the one section where the writer described instead of argued.
Where they reported instead of thinking. Where they hedged instead of committing.
Section name:
What it says now:
What it should say (one sharp sentence that takes a real position):

─────────────────────────────────────────
Q4 — RHYTHM AND NATURALNESS AUDIT
─────────────────────────────────────────
This is a pattern audit — not a single sentence fix. Read the whole blog for these four patterns.

PATTERN A — SAME-LENGTH SENTENCES:
Find any sequence of 3+ consecutive sentences at roughly the same length.
Quote the sequence. This is the most common sign of robotic writing.

PATTERN B — THROAT-CLEARING OPENERS:
Scan every paragraph opening. List every one that starts with:
"This shows...", "It is important...", "In this section...", "Now let's look at...",
"As we can see...", "With that in mind...", "There is/are...", "It is worth noting...",
"This means that...", "What this tells us...", "This demonstrates..."
Quote each one you find.

PATTERN C — SIGNIFICANCE ANNOUNCEMENTS:
Find sentences that announce their own importance instead of earning it.
Examples: "This is a crucial point.", "This reveals something significant.",
"This is key to understanding X.", "What makes this remarkable is..."
These sentences exist because the paragraph before them didn't do its job.
Quote each one.

PATTERN D — SECTION OPENINGS THAT ANNOUNCE INSTEAD OF START:
Find any section that opens by telling the reader what it's about to cover.
Examples: "In this section, we'll explore...", "To understand X, we need to first look at Y.",
"Now that we've covered X, let's turn to Y."
Quote each offending opening.

For each pattern found:
- Quote the offending text
- Write ONE rewritten version that fixes it

Overall answer: NATURAL / ROBOTIC
(ROBOTIC if any pattern found in 2+ sections. NATURAL only if the blog is clean across all four.)

─────────────────────────────────────────
Q5 — DOES THE ENDING DELIVER SOMETHING NEW?
─────────────────────────────────────────
The ending should say something the reader couldn't have predicted from the opening.
Not a summary. Not "this proves that X." A new thought.

ALSO CHECK — THESIS CONTRADICTION:
Read the title or opening premise. Now read the final paragraph.
Does the ending AGREE with the opening argument — or does it accidentally argue the opposite?

Example of a contradiction to catch:
- Title says: "Why the losses were his greatest innings"
- Ending says: "brilliance without victory is still a hollow triumph"
- That ending CONTRADICTS the thesis. The blog promised to argue losses have value,
  then concluded losses prove the opposite. Flag this as a critical failure.

Answer: NEW / SUMMARY / CONTRADICTION
If SUMMARY — what should the ending actually say? (One sentence of direction)
If CONTRADICTION — quote the specific line that contradicts the opening, and state what the ending should argue instead:

─────────────────────────────────────────
Q6 — THE ONE THING TEST
─────────────────────────────────────────
If the reader remembers ONE thing from this blog tomorrow — what will it be?
Is that the thing you WANT them to remember?
What they'll remember:
Is it the right thing: YES / NO
If NO — what should they remember instead:

─────────────────────────────────────────
Q7 — REPETITION AUDIT
─────────────────────────────────────────
This is a full scan of the blog for repeated content.
Read every section and track which matches, events, and conclusions appear more than once.

STEP 1 — LIST THE REPEATED FACTS:
Go through the blog section by section.
For each named match, innings, event, or analytical conclusion — note which sections it appears in.

Format:
- [Match/Fact]: appears in Section X and Section Y
- [Conclusion/Insight]: appears in Section X and Section Z

Only list items that appear in 2 or more sections.

STEP 2 — CLASSIFY EACH REPEAT:

For each repeated item, classify it as one of:
  DEEPENED — the second mention adds a genuinely new layer (new angle, new consequence, new dimension)
              This is acceptable. The match is being developed, not repeated.
  RESTATED  — the second mention makes the same point in different words.
              This is a problem. One instance must be cut or replaced with new content.
  RE-EXPLAINED — the second mention explains the match/event as if the reader hasn't seen it yet.
              This is the worst case. The section needs to be rewritten around a different idea.

STEP 3 — VERDICT:
Answer: CLEAN / REPETITIVE
CLEAN = no RESTATED or RE-EXPLAINED repeats found.
REPETITIVE = one or more RESTATED or RE-EXPLAINED repeats found.

If REPETITIVE — list each offending section and one sentence on what it should do instead:

─────────────────────────────────────────
OVERALL
─────────────────────────────────────────
Pass: YES or NO
(YES only if Q1=YES, Q2=SPINE, Q5=NEW, Q7=CLEAN, and at most one failure in Q3/Q4/Q6)
Priority Fix: (the single change that would most elevate this blog — be specific, one sentence)

Blog:
{_blog_for_critic}
"""
    result = _invoke(prompt, temperature=0.2)
    logger.info(f"[CRITIC] Done | blog_chars={len(blog)} | truncated={'yes' if len(blog) >= 8_000 else 'no'} | duration={time.time()-start:.1f}s")
    return result


@traceable(name="targeted_revision")
def targeted_revision(blog: str, critique: str, topic: str, audience: str) -> str:
    """
    Targeted Revision — fixes ONLY the issues flagged by the Story Critic.
    Surgical. Does not rewrite sections that passed.
    """
    _check_cancelled()
    logger.info(f"[REVISION] Starting targeted revision | topic='{topic[:60]}'")
    start = time.time()

    prompt = f"""
You are a surgical editor. Fix only what the Story Critic flagged — nothing else.

Topic: {topic}
Audience: {audience}

STORY CRITIC DIAGNOSIS:
{critique}

═══════════════════════════════════════
WHAT TO FIX
═══════════════════════════════════════

Work through the critique in this order. For each issue flagged, apply the fix below.
Skip any item the critic marked as passing.

─────────────────────────────────────────
FIX 1 — OPENING (if Q1 failed)
─────────────────────────────────────────
If the critic gave a sharper opening — use it exactly.
If not, replace the current opening using this principle:

WRONG (announces the topic):
"MS Dhoni is widely regarded as one of cricket's greatest finishers."

RIGHT (drops the reader into tension):
"Everyone remembers the six. Nobody talks about the ten chases before it."

The opening must create a gap the reader needs filled. Not introduce a subject.

─────────────────────────────────────────
FIX 2 — NARRATIVE SPINE (if Q2 = STACKED)
─────────────────────────────────────────
The critic named the section where the thread breaks.
At the END of that section, add one sentence that plants a question
the NEXT section answers. The reader should feel pulled, not pushed.

WRONG transition (announces next section):
"Now that we understand his technique, let's look at his mental approach."

RIGHT transition (creates pull):
"The technique explains how. It doesn't explain why he gets better when the pressure gets worse."

One sentence. At the end of the broken section. That's all.

─────────────────────────────────────────
FIX 3 — SAFE SECTION (if Q3 found one)
─────────────────────────────────────────
The critic named the section and the paragraph that plays it safe.
Replace the weakest paragraph with something that takes a real position.

WRONG (describes without arguing):
"Dhoni's captaincy had a significant influence on his batting approach.
His leadership experience gave him a broader perspective on the game."

RIGHT (argues, shows reasoning):
"He was the only batsman on the field who also knew where every fielder was going to be
in the next two overs. That's not confidence. That's a different kind of information."

The replacement paragraph must say something the reader wasn't expecting.
Not more information — a sharper claim.

─────────────────────────────────────────
FIX 4 — ROBOTIC PATTERNS (if Q4 = ROBOTIC)
─────────────────────────────────────────
The critic listed specific patterns found. Fix each one.

FOR SAME-LENGTH SENTENCE SEQUENCES:
Take the quoted sequence and break the rhythm.
One sentence gets cut in half. One gets extended. One becomes a fragment.

WRONG (all medium length, all same rhythm):
"He adjusted his back-lift over time. The change improved his control significantly.
His strike-rate in death overs increased as a result."

RIGHT (broken rhythm, varied length):
"He shortened his back-lift. That one adjustment — compact, almost invisible — changed
everything about how he played the death overs. His strike-rate proved it."

FOR THROAT-CLEARING OPENERS:
Delete the entire opening sentence of the offending paragraph. Start with what was sentence two.
Test: does the paragraph still make sense? It will. It will also be better.

WRONG: "This shows that the back-lift adjustment was crucial to his evolution as a finisher."
RIGHT: (delete it — the paragraph before already showed it)

FOR SIGNIFICANCE ANNOUNCEMENTS:
Delete the sentence entirely. The paragraph before it should already make the reader feel it.
If the paragraph before it doesn't — rewrite the paragraph before it, not the announcement.

FOR SECTION OPENINGS THAT ANNOUNCE:
WRONG: "To understand what made him different, we need to look at his technical evolution."
RIGHT: "His back-lift was always too high. Everyone could see it. Almost no one saw what he did about it."

─────────────────────────────────────────
FIX 5 — ENDING (if Q5 = SUMMARY or CONTRADICTION)
─────────────────────────────────────────
Delete the current ending entirely. Write a new final section: 3-5 sentences.

CRITICAL — IF Q5 = CONTRADICTION:
The ending is arguing the OPPOSITE of what the blog promised in the opening.
Re-read the title/opening premise before writing the new ending.
The new ending must COMPLETE that argument — not undermine it.

Example contradiction to fix:
- Blog premise: "the losses were his greatest innings"
- Wrong ending: "brilliance without victory is a hollow triumph" ← says losses are bad
- Right ending: something that lands WHY the losses made him, not why they diminished him

The ending must deliver something the reader could NOT have predicted from the opening —
but once they read it, it feels completely inevitable.

WRONG (summary ending):
"Dhoni's greatness as a finisher comes from his technique, his mental strength, and his experience.
These qualities combined to make him the best in the business."

RIGHT (landing something new, consistent with the opening argument):
"The six everyone remembers took two seconds. What it took to be trusted with that moment
took fifteen years. Most people got the highlight. They missed the work that made it possible."

3-5 sentences. No calls to action. No motivational wrap-up.
Final sentence is a statement. Short. It lands in silence.

─────────────────────────────────────────
FIX 6 — REPETITION (if Q7 = REPETITIVE)
─────────────────────────────────────────
The critic listed the offending sections and classified each repeat as RESTATED or RE-EXPLAINED.
Fix each one using this approach:

FOR RESTATED repeats (same conclusion said differently):
Delete the repeated sentence entirely.
The section must still make sense without it — if it doesn't, replace it with a genuinely
new observation about the same subject.

WRONG: Section 3 ends with "These knocks reveal his quality under pressure."
       Section 4 ends with "His ability to perform when it matters is what defines him."
RIGHT: Keep the sharper one. Delete the other. If Section 4 needs an ending,
       write one that says something Section 3 didn't — a new consequence, a new dimension.

FOR RE-EXPLAINED repeats (same match described as if for the first time):
The section is built around an event the reader already knows.
Rewrite the section's opening so it starts with what is NEW about this angle —
not with the event itself.

WRONG: "In Adelaide, India needed 364 to win. Kohli walked in at 54/2 and scored 141..."
       (reader has already been told this — this is a re-explanation)
RIGHT: Start with the new angle: "What that Adelaide innings proved about his captaincy
       was separate from what it proved about his batting..."
       The event is referenced in one clause. The new idea gets the paragraph.

RULE: After fixing each repetition, the same match or conclusion must not appear
as the main subject of two different sections in the final blog.

═══════════════════════════════════════
HARD RULES
═══════════════════════════════════════
- Touch ONLY what the critique flagged. Leave everything else exactly as it is.
- Do NOT rewrite sections that passed.
- Do NOT change any facts, numbers, or CLAIM-IDs.
- Match the voice of the surrounding text in every fix.
- Section headings stay as ### prefix.
- Return the FULL revised blog — no commentary, no labels, no explanation.

Original Blog:
{blog}
"""
    result = _invoke(prompt, temperature=0.5)
    logger.info(f"[REVISION] Done | duration={time.time()-start:.1f}s | words={len(result.split())}")
    return result
def score_blog(blog: str, topic: str, seo: dict) -> dict:
    """
    Rubric-based scorer — LLM answers specific YES/NO/PARTIAL questions,
    Python converts answers to scores. Removes the LLM number-picking bias
    that was producing systematically pessimistic scores (5.2, 5.6).

    Scoring map per criterion:
      YES     → 9.0
      PARTIAL → 7.0
      NO      → 5.0

    Overall = weighted average of 5 criteria.
    Verdict = one-sentence summary from LLM.
    """
    import re as _re

    prompt = f"""
You are evaluating a blog about: "{topic}"

Answer each question with YES, PARTIAL, or NO — then one sentence of evidence.
Be honest but fair. PARTIAL means "it tries but doesn't fully land."

Q1 — HUMAN VOICE
Does the blog read like a person with a real point of view, not like filler content?
Answer: YES / PARTIAL / NO
Evidence: (one specific sentence from the blog that proves your answer, or explains what's missing)

Q2 — OPENING HOOK
Do the first 3 sentences make a smart reader stop and keep reading?
Answer: YES / PARTIAL / NO
Evidence: (quote the opening or name exactly what kills it)

Q3 — ORIGINAL INSIGHT
Is there at least one idea a reader couldn't get from the first Google result on this topic?
Something that reframes, surprises, or goes one level deeper?
Answer: YES / PARTIAL / NO
Evidence: (quote the insight or say what's missing)

Q4 — ENDING QUALITY
Does the final paragraph say something NEW and land with impact?
(Not a summary, not a CTA — something that makes the reader feel or think differently)
Answer: YES / PARTIAL / NO
Evidence: (quote the ending or describe what's wrong with it)

Q5 — SHAREABILITY
Would someone who cares about this topic forward this blog to a friend?
Answer: YES / PARTIAL / NO
Evidence: (one honest sentence on why or why not)

VERDICT: (one sentence — what this blog does best and the single change that would push it higher)

Blog:
{blog[:2_000]}
...
{blog[-1_500:]}

Return in this exact format — no extra lines:
Q1: YES/PARTIAL/NO | evidence
Q2: YES/PARTIAL/NO | evidence
Q3: YES/PARTIAL/NO | evidence
Q4: YES/PARTIAL/NO | evidence
Q5: YES/PARTIAL/NO | evidence
VERDICT: sentence
"""

    raw = _invoke(prompt, temperature=0.1)

    # ── Parse LLM rubric answers ──────────────────────────────────────────────
    SCORE_MAP = {"YES": 9.0, "PARTIAL": 7.0, "NO": 5.0}
    LABELS = {
        "Q1": "CRITERION 1 — HUMAN VOICE",
        "Q2": "CRITERION 2 — OPENING STRENGTH",
        "Q3": "CRITERION 3 — ONE ORIGINAL IDEA",
        "Q4": "CRITERION 4 — ENDING QUALITY",
        "Q5": "CRITERION 5 — WOULD YOU SHARE IT",
    }

    scores: dict[str, str] = {}
    numeric: list[float] = []
    verdict = ""

    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # VERDICT line
        if line.upper().startswith("VERDICT:"):
            verdict = line.partition(":")[2].strip()
            continue

        # Q1-Q5 lines: "Q1: YES | evidence sentence"
        m = _re.match(r'^(Q[1-5])\s*:\s*(YES|PARTIAL|NO)\s*[|]\s*(.+)$', line, _re.IGNORECASE)
        if m:
            key   = m.group(1).upper()
            answer = m.group(2).upper()
            evidence = m.group(3).strip()
            numeric_score = SCORE_MAP.get(answer, 7.0)
            numeric.append(numeric_score)
            label = LABELS.get(key, key)
            scores[f"Score: {label}"] = f"{numeric_score}/10"
            scores[f"Reason: {label}"] = f"{answer} — {evidence}"

    # ── Python computes overall score — no LLM number-picking ────────────────
    if numeric:
        overall = round(sum(numeric) / len(numeric), 1)
    else:
        # Fallback: couldn't parse — default to 7.0
        overall = 7.0
        logger.warning("[SCORER] Could not parse rubric answers — defaulting to 7.0")

    scores["Overall Score"] = f"{overall}/10"
    scores["Verdict"] = verdict or "Blog evaluated successfully."

    logger.info(f"[SCORER] overall={overall} | criteria={numeric}")
    return scores


def generate_extras(blog: str, topic: str, facts: str = "") -> dict:
    # Build a grounded facts reference for TL;DR so bullets don't echo hallucinated claims
    facts_ref = ""
    if facts and facts.strip():
        facts_ref = f"""
VERIFIED FACTS REFERENCE (TL;DR bullets must only use claims from here or general observations):
{facts[:2000]}
"""

    prompt = f"""
You are an editorial designer. Given this blog, generate three things:

1. TL;DR — exactly 3 bullet points summarizing the most important takeaways. Each bullet is one sharp sentence.
   RULE: Only include specific numbers or statistics if they appear in the VERIFIED FACTS REFERENCE below.
   If a number is not verified, describe the insight without the number.
   No fluff.
2. Pull Quote — pick the single most powerful, quotable sentence from the blog. It should be something that makes a reader stop and think.
3. Key Takeaway — one bold, specific sentence that captures the core message of the entire blog. Not a summary — a conclusion that lands.

Topic: {topic}
{facts_ref}

Blog:
{blog}

Return in this exact format:
TL;DR:
- [bullet 1]
- [bullet 2]
- [bullet 3]
Pull Quote: [the sentence]
Key Takeaway: [the sentence]
"""
    raw = _invoke(prompt, temperature=0.3)
    extras = {"tldr": [], "pull_quote": "", "key_takeaway": ""}

    lines = raw.strip().split("\n")
    current_section = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        lo = stripped.lower()

        # Detect section headers — LLM sometimes puts content on same line, sometimes next line
        if lo.startswith("tl;dr") or lo.startswith("tldr"):
            current_section = "tldr"
            # Content might be on the same line after the colon
            after = stripped.partition(":")[2].strip()
            if after and after.startswith("- "):
                extras["tldr"].append(after[2:].strip())
            continue

        if lo.startswith("pull quote"):
            current_section = "pull_quote"
            after = stripped.partition(":")[2].strip()
            if after:
                extras["pull_quote"] = after.strip('"').strip("'").strip()
            continue

        if lo.startswith("key takeaway"):
            current_section = "key_takeaway"
            after = stripped.partition(":")[2].strip()
            if after:
                extras["key_takeaway"] = after.strip('"').strip("'").strip()
            continue

        # Content lines — assign to current section
        if current_section == "tldr":
            if stripped.startswith("- "):
                extras["tldr"].append(stripped[2:].strip())
            elif stripped.startswith("* "):
                extras["tldr"].append(stripped[2:].strip())
            elif stripped[0].isdigit() and stripped[1] in ".):":
                extras["tldr"].append(stripped[2:].strip())
            # If pull quote or key takeaway section starts, current_section changes above

        elif current_section == "pull_quote":
            if not extras["pull_quote"]:
                extras["pull_quote"] = stripped.strip('"').strip("'").strip()

        elif current_section == "key_takeaway":
            if not extras["key_takeaway"]:
                extras["key_takeaway"] = stripped.strip('"').strip("'").strip()

    # ── Fallback: retry with a simpler prompt if TL;DR is still empty ────────
    # This handles cases where the LLM returned a non-standard format
    if not extras["tldr"]:
        logger.warning("[EXTRAS] TL;DR empty after parse — retrying with simpler prompt")
        try:
            fallback_raw = _invoke(f"""
Summarize this blog about "{topic}" in exactly 3 bullet points.
Each bullet must be one sharp, specific sentence.
Return ONLY the 3 lines, each starting with "- ".

Blog:
{blog[:3_000]}
""", temperature=0.3)
            for line in fallback_raw.strip().split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    extras["tldr"].append(line[2:].strip())
                elif line.startswith("* "):
                    extras["tldr"].append(line[2:].strip())
            logger.info(f"[EXTRAS] Fallback TL;DR produced {len(extras['tldr'])} bullets")
        except Exception as e:
            logger.warning(f"[EXTRAS] Fallback TL;DR failed: {e}")

    # ── Fallback: pull quote if still empty ───────────────────────────────────
    if not extras["pull_quote"]:
        logger.warning("[EXTRAS] Pull quote empty — using last sentence of blog as fallback")
        import re as _re
        sentences = _re.split(r'(?<=[.!?])\s+', blog.strip())
        clean = [s.strip() for s in sentences if len(s.strip()) > 40 and not s.strip().startswith("#")]
        if clean:
            extras["pull_quote"] = clean[-1]

    return extras


def generate_citations(blog: str, facts: str, topic: str, research_data: str = "") -> str:
    """
    Programmatic citation builder — no LLM, no invented titles.
    Parses raw_facts (format: [N] claim | source_title | source_url)
    and builds the Sources section directly from real data.
    Then adds inline [N] markers to the blog where claims appear.

    Always appends a Sources block when facts are available — even when
    inline matching fails (paraphrased content, broad topics, etc.)
    """
    import re

    if not facts or not facts.strip():
        logger.warning("[CITATIONS] Empty facts string — skipping citations")
        return blog

    # Parse the structured facts list
    source_map = {}  # citation_num -> (claim, title, url)
    # Deduplicate by URL so the same source doesn't appear multiple times
    seen_urls: set[str] = set()

    for line in facts.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Format: [N] claim | source_title | source_url
        m = re.match(r'^\[(\d+)\]\s+(.+?)\s+\|\s+(.+?)\s+\|\s+(https?://\S+)', line)
        if m:
            num, claim, title, url = m.group(1), m.group(2), m.group(3), m.group(4)
            url = url.rstrip(".,;)")  # strip trailing punctuation that breaks links
            if url not in seen_urls:
                seen_urls.add(url)
                source_map[num] = (claim.strip(), title.strip(), url.strip())

    if not source_map:
        logger.warning("[CITATIONS] No structured facts parsed — skipping citations")
        return blog

    # ── Inline citation markers ───────────────────────────────────────────────
    # For each fact, find its key words in the blog and append [N] after the sentence.
    # Uses a relaxed word-overlap heuristic so paraphrased text still matches.
    cited_blog = blog
    used_citations: list[str] = []

    for num, (claim, title, url) in source_map.items():
        # Extract the most distinctive keywords from the claim (skip stopwords)
        stopwords = {"that", "this", "with", "from", "have", "been", "were", "their", "they"}
        claim_words = [w for w in claim.lower().split() if len(w) > 3 and w not in stopwords][:6]
        if not claim_words:
            continue

        # Try matching with progressively relaxed thresholds
        matched = False
        for anchor in claim_words[:3]:  # try up to 3 anchor words
            pattern = r'([^.!?]*' + re.escape(anchor) + r'[^.!?]*[.!?])'
            matches = list(re.finditer(pattern, cited_blog, re.IGNORECASE))
            for match in matches:
                sentence = match.group(0)
                words_found = sum(1 for w in claim_words if w in sentence.lower())
                # Relaxed: 1 out of N is enough when claim is short, 2+ for longer claims
                threshold = 1 if len(claim_words) <= 2 else 2
                if words_found >= threshold:
                    marker = f"[{num}]"
                    end_pos = match.end()
                    # Avoid double-marking the same spot
                    context = cited_blog[max(0, end_pos - 25):end_pos + 5]
                    if marker not in context:
                        cited_blog = cited_blog[:end_pos - 1] + marker + cited_blog[end_pos - 1:]
                        used_citations.append(num)
                        matched = True
                        break
            if matched:
                break

    # ── Build Sources block with sequential renumbering ───────────────────────
    # Collect final display order: inline-cited sources first, then remaining.
    # Then renumber everything 1, 2, 3... so the reader never sees gaps like
    # [1], [5], [13], [18].
    seen_shown: set[str] = set()
    ordered_sources: list[tuple[str, str, str]] = []  # (old_num, title, url)

    # Inline-cited sources first (preserving order)
    for num in used_citations:
        if num not in seen_shown and num in source_map:
            _, title, url = source_map[num]
            ordered_sources.append((num, title, url))
            seen_shown.add(num)

    # Remaining sources that weren't matched inline
    for num, (_, title, url) in source_map.items():
        if num not in seen_shown:
            ordered_sources.append((num, title, url))
            seen_shown.add(num)

    # Build old_num → new_num mapping for renumbering inline markers
    old_to_new: dict[str, str] = {}
    sources_lines = ["Sources"]
    for new_idx, (old_num, title, url) in enumerate(ordered_sources, 1):
        old_to_new[old_num] = str(new_idx)
        sources_lines.append(f"[{new_idx}] {title} — {url}")

    # Replace inline markers in blog body: [5] → [2] etc.
    # Sort by length descending to avoid partial replacements (e.g. [1] inside [13])
    for old_num in sorted(old_to_new.keys(), key=lambda x: -len(x)):
        new_num = old_to_new[old_num]
        if old_num != new_num:
            cited_blog = cited_blog.replace(f"[{old_num}]", f"[{new_num}]")

    logger.info(
        f"[CITATIONS] inline_markers={len(used_citations)} | "
        f"total_sources={len(sources_lines) - 1}"
    )

    sources_block = "\n".join(sources_lines)
    return cited_blog.rstrip() + "\n\n" + sources_block


@traceable(name="generate_seo")
def generate_seo(topic: str, blog: str) -> dict:
    word_count = len(blog.split())
    read_time = max(1, round(word_count / 200))

    prompt = f"""
You are an SEO specialist writing for both search engines and real humans.

Blog topic: "{topic}"

Generate in plain text (no hashtags, no emojis, no numbering):

Meta Title: (under 60 characters — specific and compelling, not just the topic name)
Meta Description: (under 160 characters — tell the reader exactly what value they get)
Focus Keyword: (single most searchable keyword phrase)
Secondary Keywords: (5 long-tail keyword phrases, comma separated)
Suggested Tags: (5 to 7 tags, comma separated)

Blog:
{blog}

Return each on its own line with label followed by colon. Nothing else.
"""
    raw = _invoke(prompt, temperature=0.3)
    seo = {}
    for line in raw.strip().split("\n"):
        if ":" in line:
            key, _, value = line.partition(":")
            seo[key.strip()] = value.strip().rstrip(":")
    seo["Estimated Read Time"] = f"{read_time} min read"
    return seo
