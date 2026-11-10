import os  #read env variables like API Keys
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
from research_engine import run_research_engine


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
]

LENGTH_GUIDE = {
    "short": "600 to 800 words. 3 to 4 sections. Every sentence must earn its place. Dense, sharp, zero filler. Short does not mean shallow — pack maximum insight into minimum words.",
    "medium": "1000 to 1500 words. 5 to 6 sections. Each section goes one level deeper than the surface. At least 2 specific data points or named examples per section.",
    "long": "2000 to 2500 words. 7 to 8 sections. Full depth on every angle. Multiple named examples, data points, and at least 2 strong analogies throughout. Every section should feel like it could stand alone."
}


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

STEP 1 — TOPIC INTELLIGENCE
Topic Type: (one of) Technical | Sports | Educational | Historical | Analytical | Review | Opinion | Science | Health | Business
Core Question: The single most important question this blog must answer.
Reader Pain Point: What does the reader actually want to understand or solve?
Unique Angle: What do most articles on this topic miss? What makes this one different?

STEP 2 — WRITING APPROACH
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

STEP 3 — STRUCTURE
Opening Hook: (specific — a scene, stat, contradiction or uncomfortable truth for THIS topic)
Sections: (5-7 headings, each with one line on what it must prove)
Core Argument: (the single insight the reader leaves with)
Conclusion: (how to land it — challenge, reframe, hard truth or memorable final thought)

STEP 4 — TONE AND VOICE
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
def research(topic: str, competitor_gap: str = "") -> tuple[str, str]:
    """
    Full 6-phase research engine.
    Returns (dossier_str, raw_facts_str).
    dossier_str   — structured evidence dossier for write_blog
    raw_facts_str — flat facts list for generate_citations
    """
    _check_cancelled()
    logger.info(f"[RESEARCH] Starting research engine | topic='{topic[:60]}'")
    start = time.time()
    dossier, raw_facts = run_research_engine(topic, competitor_gap, _invoke)
    logger.info(f"[RESEARCH] Done | duration={time.time()-start:.1f}s")
    return dossier, raw_facts


@traceable(name="extract_facts")
def extract_facts(topic: str, research_data: str) -> str:
    """
    Backward-compatible shim — raw_facts_str is already built by the research engine.
    This just passes it through so graph.py nodes need minimal changes.
    """
    return research_data


@traceable(name="unified_writer")
def write_blog(topic: str, audience: str, plan: str, research_data: str, memory: str, length: str = "medium", gap: str = "", facts: str | None = None) -> str:
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

    prompt = f"""
You are a writer who has internalized three masters. You apply them adaptively — different topics demand different emphases.

Topic: {topic}
Target Audience: {audience}
Length: {length_instruction}

WRITING STRATEGY (follow this exactly):
{plan_input}
{gap_section}

RESEARCH DOSSIER — YOUR ONLY SOURCE OF FACTS:
{verified_facts_input}

PAST CONTEXT (from memory):
{memory}

ABSOLUTE FACT RULES — violating these makes the blog unpublishable:
1. Every number, percentage, statistic in your blog MUST exist word-for-word in the dossier above.
2. If a number is NOT in the dossier — do not write it. Write the insight without the number instead.
3. CONFIRMED FACTS: state directly and confidently.
4. SINGLE-SOURCE FACTS: attribute — "according to [source name]..."
5. DISPUTED: write "some sources suggest X, while others report Y"
6. Never invent a year, match, event or measurement not in the dossier.

ADAPTIVE MECHANIC EMPHASIS FOR THIS TOPIC ({topic_type.upper()}):
Primary (lead with this): {primary}
Secondary (support with this): {secondary}
Light use (one or two moments): {light}

THE THREE MECHANICS — full toolkit:

PAUL GRAHAM MECHANICS:
1. Uncomfortable Truth Opening — something true that most people sense but haven't said out loud
2. Reasoning Chain — show the steps, not just the conclusion. "If X is true, then Y follows. But most people stop at X."
3. Contrarian Insight — the thing that seems obvious but isn't. Makes the reader think "I never thought about it that way."
4. The "Why does this actually work?" question — ask it and answer it with real reasoning, not platitudes
5. No authority without explanation — never cite someone without explaining WHY they're right

TIM URBAN MECHANICS:
1. Zoom Out Opening — show the scale of what the reader is about to understand before diving in
2. Relatable Villain — name the exact mechanism causing the problem, not "the system" or "society"
3. Bridge Analogy — one per major concept. Physical, visual, specific. Not "it's like a machine" — "it's like a bicycle gear: same pedaling, completely different speed."
4. Curiosity Loop — end every section with a statement that creates tension and pulls toward the next
5. The "You" Frame — address the reader's actual situation, not a generic person

GEORGE ORWELL MECHANICS:
1. One-Sentence Test — every sentence must do work no other sentence does. Cut anything that doesn't.
2. Concrete over Abstract — "saved 3 hours" not "increased efficiency". "fewer emails" not "improved communication"
3. Active Voice — "the study found" not "it was found". Passive only when doer is genuinely unknown
4. Strong Verbs — cuts, doubles, eliminates, forces, reveals. Not "helps improve" or "can be used to"
5. Sentence Rhythm — short for impact. Longer for flow. Never three of the same length in a row.
6. Banned words — never use: utilize, leverage, impactful, holistic, synergy, paradigm, "it's important to note", "in today's world", "needless to say", "at the end of the day", "game changer"

HOW THEY COMBINE (regardless of emphasis):
- Opening: zoom out to show scale (Urban) + land the uncomfortable truth (Graham) — first 2-3 sentences
- Each section body: reasoning chain (Graham) + concrete precision (Orwell)
- Each section's analogy: one bridge analogy (Urban) — one per major concept minimum
- Each section's close: curiosity loop sentence (Urban)
- Conclusion: one original insight (Graham) + sharpest possible final sentence (Orwell)

STRUCTURE RULES:
- Section headings: ### before each heading, nothing else
- Short paragraphs — 2 to 4 lines maximum
- Blank line between every paragraph and section
- No emojis, no hashtags, no bullet points in the main body

Write the full blog now. Lead with {primary.split('—')[0].strip()}. Every word earns its place.
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
        # Check if the raw number appears anywhere in the dossier
        if num.lower() not in dossier_lower:
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
- Remove just the number/statistic that is in the unverified list
- Keep the rest of the sentence intact and natural
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

    # ── Apply fixes back to the blog ──────────────────────────────────────────
    corrected_blog = blog
    for block in raw.strip().split("\n\n"):
        lines = block.strip().split("\n")
        orig_line = next((l[len("ORIGINAL:"):].strip() for l in lines if l.startswith("ORIGINAL:")), None)
        fixed_line = next((l[len("FIXED:"):].strip() for l in lines if l.startswith("FIXED:")), None)
        if orig_line and fixed_line and orig_line != fixed_line:
            corrected_blog = corrected_blog.replace(orig_line, fixed_line, 1)

    logger.info(f"[AUDIT] Done | duration={time.time()-start:.1f}s")
    return corrected_blog


@traceable(name="editorial_critic")
def critique_and_rewrite(blog: str, topic: str, audience: str) -> str:
    """
    Editorial Critic — evaluates the blog and returns a structured critique.
    Does NOT rewrite. Quality Router reads this output to decide if revision needed.
    If revision needed, targeted_revision fixes only the flagged issues.
    """
    _check_cancelled()
    logger.info(f"[CRITIC] Starting | topic='{topic[:60]}'")
    start = time.time()

    prompt = f"""
You are a senior editor who evaluates blogs against three specific writing mechanics: Paul Graham (original insight), Tim Urban (storytelling and curiosity), George Orwell (clarity and precision).

Topic: {topic}
Target Audience: {audience}

Evaluate this blog on 5 dimensions. Score 1-10 and give ONE specific fix if needed.

DIMENSION 1 — OPENING STRENGTH
Does it open with an uncomfortable truth or zoom-out? Does it hook by line 3?
Score: X/10
Issue: (specific weakness or "None")
Fix: (exact change or "None")

DIMENSION 2 — ORIGINAL INSIGHT (Paul Graham)
Does each section have one insight the reader couldn't have Googled? Is the reasoning shown not just stated?
Score: X/10
Issue: (which section is just summarizing, or "None")
Fix: (what original angle to add, or "None")

DIMENSION 3 — ANALOGIES AND CURIOSITY (Tim Urban)
Is there at least one strong bridge analogy? Does each section make the reader want the next?
Score: X/10
Issue: (missing analogy or weak curiosity loop, or "None")
Fix: (specific analogy to add or section to close better, or "None")

DIMENSION 4 — CLARITY AND PRECISION (Orwell)
Are there abstract nouns that should be concrete? Passive voice? Banned words? Weak verbs?
Score: X/10
Issue: (specific sentences that fail, or "None")
Fix: (rewrite those specific sentences, or "None")

DIMENSION 5 — CONCLUSION
Is it a new thought that lands, or a recycled summary?
Score: X/10
Issue: (specific weakness or "None")
Fix: (what the conclusion should say instead, or "None")

OVERALL VERDICT
Pass (all dimensions >= 7): YES or NO
Priority Fix: (if NO — the single most important fix)

Blog:
{blog}
"""
    result = _invoke(prompt, temperature=0.2)
    logger.info(f"[CRITIC] Done | duration={time.time()-start:.1f}s")
    return result


@traceable(name="targeted_revision")
def targeted_revision(blog: str, critique: str, topic: str, audience: str) -> str:
    """
    Targeted Revision — fixes ONLY the issues flagged by the Editorial Critic.
    Does not rewrite the whole blog. Surgical and fast.
    """
    _check_cancelled()
    logger.info(f"[REVISION] Starting targeted revision | topic='{topic[:60]}'")
    start = time.time()

    prompt = f"""
You are a precise editor. You fix only what is broken — nothing else.

Topic: {topic}
Target Audience: {audience}

EDITORIAL CRITIQUE (fix only what is flagged here):
{critique}

RULES:
- Fix ONLY the specific issues listed in the critique above
- Do NOT rewrite sections that scored 7 or above
- Do NOT change the structure unless the critique explicitly flags it
- Do NOT change facts, numbers or citations
- Keep the same voice, tone and style throughout
- Section headings must stay as ### prefix
- If the critique says "None" for an issue — leave that section completely untouched

Return the full revised blog. No commentary.

Original Blog:
{blog}
"""
    result = _invoke(prompt, temperature=0.5)
    logger.info(f"[REVISION] Done | duration={time.time()-start:.1f}s | words={len(result.split())}")
    return result
def score_blog(blog: str, topic: str, seo: dict) -> dict:
    prompt = f"""
You are a professional blog quality analyst. Score objectively. Be honest, not generous.

Topic: {topic}

Blog:
{blog}

SEO Metadata:
{seo}

Score each from 1 to 10 with a one-line reason:

Readability: (short clear sentences? easy to follow?)
Hook Strength: (does the opening grab immediately with something specific?)
Content Depth: (real data, named examples, specific facts?)
SEO Strength: (focus keyword used naturally? meta title compelling?)
Conclusion Quality: (memorable and specific, not a cliché?)

Overall Score: (average of the 5, out of 10)
Verdict: (one sharp sentence — what it does well and what would push it higher)

Return each on its own line with label followed by colon. Nothing else.
"""
    raw = _invoke(prompt, temperature=0.0)
    scores = {}
    for line in raw.strip().split("\n"):
        if ":" in line:
            key, _, value = line.partition(":")
            scores[key.strip()] = value.strip()
    return scores


def generate_extras(blog: str, topic: str) -> dict:
    prompt = f"""
You are an editorial designer. Given this blog, generate three things:

1. TL;DR — exactly 3 bullet points summarizing the most important takeaways. Each bullet is one sharp sentence. No fluff.
2. Pull Quote — pick the single most powerful, quotable sentence from the blog. It should be something that makes a reader stop and think.
3. Key Takeaway — one bold, specific sentence that captures the core message of the entire blog. Not a summary — a conclusion that lands.

Topic: {topic}

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

    # ── Build Sources block ───────────────────────────────────────────────────
    # Always show ALL sources — inline-cited ones first, then remaining.
    # This ensures the Sources block is never missing even when inline
    # matching fails completely (e.g. heavily paraphrased or broad topics).
    seen_shown: set[str] = set()
    sources_lines = ["Sources"]

    # Inline-cited sources first (preserving order)
    for num in used_citations:
        if num not in seen_shown and num in source_map:
            _, title, url = source_map[num]
            sources_lines.append(f"[{num}] {title} — {url}")
            seen_shown.add(num)

    # Remaining sources that weren't matched inline
    for num, (_, title, url) in source_map.items():
        if num not in seen_shown:
            sources_lines.append(f"[{num}] {title} — {url}")
            seen_shown.add(num)

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
