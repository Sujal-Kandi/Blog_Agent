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
def generate_opening_hook(topic: str, insights: str, plan: str, audience: str) -> str:
    """
    Opening Hook Generator — one dedicated LLM call, one job.

    Generates 5 candidate opening sentences then picks the sharpest one.
    The opening is the hardest sentence to write and the most important.
    When buried inside a 2000-word prompt the LLM rushes it.
    A dedicated call with a single focus produces a genuinely sharp first sentence.
    """
    _check_cancelled()
    logger.info(f"[HOOK] Generating opening hook | topic='{topic[:60]}'")
    start = time.time()

    plan_input     = plan[:1_000]    if plan     else ""
    insights_input = insights[:1_000] if insights else ""

    prompt = f"""
You are writing the opening sentence of a blog about: "{topic}"
Target audience: {audience}

CORE INSIGHTS TO DRAW FROM:
{insights_input}

STRATEGY BRIEF:
{plan_input}

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
REASON: [one sentence on why this one wins for this topic and audience]
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
        skip_starts = ("type ", "chosen", "reason", "---", "hook ")
        candidates = [
            l.strip().strip('"').strip("'")
            for l in result.split("\n")
            if l.strip()
            and len(l.strip()) > 15
            and len(l.strip()) < 200
            and not any(l.strip().lower().startswith(s) for s in skip_starts)
        ]
        chosen = candidates[-1] if candidates else result[:200]

    logger.info(f"[HOOK] Done | hook='{chosen[:80]}' | duration={time.time()-start:.1f}s")
    return chosen


@traceable(name="ending_generator")
def generate_ending(blog: str, topic: str, insights: str) -> str:
    """
    Ending Generator — runs after the draft is written, before citations.
    Returns ONLY the clean final paragraph text — no labels, no metadata.
    """
    _check_cancelled()
    logger.info(f"[ENDING] Generating ending | topic='{topic[:60]}'")
    start = time.time()

    blog_tail      = blog[-1_500:]  if blog     else ""
    insights_input = insights[:800] if insights else ""

    prompt = f"""
You are writing the final paragraph of a blog about: "{topic}"

CORE INSIGHTS:
{insights_input}

CURRENT ENDING (last section of blog):
{blog_tail}

Write 3 candidate endings (labeled A, B, C). Each: 2-4 sentences, says something NEW, lands hard.
A = REFRAME: shows the topic means something bigger than assumed
B = CHALLENGE: asks reader to do or think something specific right now
C = HARD TRUTH: uncomfortable conclusion that follows from the blog

After the 3 options write exactly this line: ---CHOSEN---
Then write ONLY the text of the best ending. No labels. No "FINAL TEXT:". Just the sentences.
"""
    result = _invoke(prompt, temperature=0.6)

    # Extract everything after ---CHOSEN---
    if "---CHOSEN---" in result:
        final_text = result.split("---CHOSEN---", 1)[1].strip()
    else:
        # Fallback: take the last clean paragraph that isn't a label
        skip_starts = ("ending a", "ending b", "ending c", "chosen",
                       "final text", "---", "option", "reframe", "challenge", "hard truth")
        paragraphs = [p.strip() for p in result.split("\n\n") if p.strip()]
        clean = [p for p in paragraphs
                 if not any(p.lower().startswith(s) for s in skip_starts)]
        final_text = clean[-1] if clean else ""

    # Strip any remaining label lines that slipped through
    skip_prefixes = ("ending a", "ending b", "ending c", "chosen ending",
                     "final text", "---", "option a", "option b", "option c",
                     "reframe:", "challenge:", "hard truth:")
    clean_lines = [
        line for line in final_text.split("\n")
        if not any(line.strip().lower().startswith(p) for p in skip_prefixes)
    ]
    final_text = "\n".join(clean_lines).strip()

    logger.info(f"[ENDING] Done | chars={len(final_text)} | duration={time.time()-start:.1f}s")
    return final_text


@traceable(name="unified_writer")
def write_blog(topic: str, audience: str, plan: str, research_data: str, memory: str, length: str = "medium", gap: str = "", facts: str | None = None, hook: str = "", insights: str = "") -> str:
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

    # Voice profile per topic type — makes every blog feel natural to its subject
    VOICE_PROFILES = {
        "sports":      "You are a passionate, knowledgeable cricket/sports fan who has watched thousands of hours of this sport. You talk like someone who actually loves the game — not a journalist filing a report, not a coach writing a manual. You have opinions. You get excited. You make comparisons a true fan would make.",
        "technical":   "You are a sharp engineer or practitioner who has worked in this field. You explain things the way a great mentor does — not dumbing it down, not showing off. You know which details actually matter and which ones are noise.",
        "educational": "You are a naturally curious person who just went deep on this topic and can't stop thinking about it. You explain it like you're telling a smart friend what you learned — genuinely excited, not performing excitement.",
        "historical":  "You are a historian who makes the past feel alive and relevant. You connect what happened then to what it means now. You have a point of view — history isn't just dates and names to you.",
        "analytical":  "You are a sharp thinker who has thought about this more carefully than most people. You're willing to say something uncomfortable if the evidence supports it. You show your reasoning, not just your conclusions.",
        "opinion":     "You are someone with a clear, well-reasoned position on this topic. You acknowledge the other side honestly before dismantling it. You don't hedge. You say what you actually think.",
        "science":     "You are a science journalist who bridges the gap between research papers and real people. You make complex things concrete. You care about accuracy but hate jargon. You find the human story inside the data.",
        "health":      "You are a health writer who respects both the science and the reader's intelligence. No fear-mongering, no oversimplification. You tell people what the evidence actually says and what it means for their life.",
        "business":    "You are a sharp business thinker with real experience. You cut through the management-speak. You talk about money, incentives, and decisions honestly — the way good operators do in private.",
        "review":      "You are a trusted critic who has actually used/seen/experienced what you're reviewing. You have a clear verdict and you defend it with specific evidence. You are honest even when it's uncomfortable.",
        "concept_explainer": "You are a thoughtful person who finds this idea genuinely fascinating and wants to share that fascination. You don't lecture. You explore. You let the reader discover alongside you.",
        "person_biography":  "You are writing about someone you genuinely find interesting. You have a perspective on what makes them remarkable — not just their achievements, but the specific thing about how they think or operate that most people miss.",
        "historical_event":  "You bring history alive by putting the reader in the room. You care about why things happened, not just what happened. You connect the past to something the reader can feel today.",
    }

    voice = VOICE_PROFILES.get(topic_type, "You are a smart, curious writer who has thought deeply about this topic. You write with a clear point of view — not neutral, not academic. Human.")

    # Opening hook section — if a pre-generated hook exists, the writer leads with it
    hook_section = ""
    if hook and hook.strip():
        hook_section = f"""
OPENING HOOK — START WITH THIS EXACT SENTENCE:
{hook.strip()}

Do not change it. Do not add "Introduction" before it. The blog starts with this sentence.
"""

    # Insights section — the pre-extracted ideas for the writer to build around
    insights_section = ""
    if insights and insights.strip():
        insights_section = f"""
CORE INSIGHTS TO BUILD THE BLOG AROUND:
{insights[:2_000]}

One of these should be the spine of the whole blog. Pick the one that fits best and develop it deeply.
Don't just mention all three and move on — choose one as the central argument and let the others support it.
"""

    prompt = f"""
{voice}

Your job: write a blog that a real person actually wants to read. Not a report. Not a Wikipedia summary. Not a coaching manual. A blog — something with a voice, a point of view, and a reason to exist.

Topic: {topic}
Target Audience: {audience}
Length: {length_instruction}

WRITING STRATEGY:
{plan_input}
{gap_section}
{hook_section}
{insights_section}

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
ADAPTIVE MECHANICS: {topic_type.upper()}
═══════════════════════════════════════

Primary: {primary}
Secondary: {secondary}
Light use: {light}

PAUL GRAHAM: uncomfortable truth upfront → show the reasoning chain, not just the conclusion → contrarian reframe → "why does this actually work?" answered with real logic
TIM URBAN: zoom out to show scale → one relatable villain/mechanism → bridge analogy (physical, visual, specific — not "it's like a machine") → curiosity loop at section end → "you" frame
ORWELL: every sentence earns its place → concrete over abstract → active voice → strong verbs → rhythm (never three same-length sentences in a row) → no banned words

BANNED WORDS: utilize, leverage, impactful, holistic, synergy, paradigm, "it's important to note", "in today's world", "needless to say", "at the end of the day", "game changer", "deep dive", "journey"

═══════════════════════════════════════
STRUCTURE
═══════════════════════════════════════

- Short paragraphs — 2 to 4 lines max. Blank line between paragraphs.
- Section headings: ### before each heading
- No bullet points or numbered lists in the main body
- No emojis, no hashtags
- Vary what sections DO: some explain, some tell a story, some argue, some zoom out, some zoom in
- Do NOT repeat the same structural element at the end of every section

═══════════════════════════════════════
FACT RULES
═══════════════════════════════════════

1. NEVER invent a specific match, game, date, score or event not in the dossier.
   WRONG: "In the Eden Gardens Test on day four, Rohit faced a full-toss..."
   RIGHT: "When Rohit faces a full-toss, his back foot is already lifting..."
   One fakes a source. The other describes a real pattern honestly.

2. Every specific NUMBER must come from the dossier. If not there — write the insight without the number.

3. If the dossier is thin — write a richer blog using analysis, pattern recognition, and opinion.
   A great blog does not need 20 statistics. It needs one original idea developed deeply.

═══════════════════════════════════════
THE ONE TEST
═══════════════════════════════════════

Before writing each section ask: would a passionate fan of this topic learn something, feel something, or see something differently after reading this?

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
    Human Rubric Critic — evaluates the blog the way a real reader would.
    Asks the questions that matter: would you share this? does it have an opinion?
    is there one sentence you'd quote? did the opening earn attention?
    """
    _check_cancelled()
    logger.info(f"[CRITIC] Starting | topic='{topic[:60]}'")
    start = time.time()

    prompt = f"""
You are a sharp, honest editor. You read this blog the way a real person would — not checking boxes, actually reading it.

Topic: {topic}
Audience: {audience}

Answer these 5 questions about the blog. Be specific and brutally honest.

QUESTION 1 — WOULD YOU SHARE THIS?
If a friend who cares about this topic sent you this blog, would you forward it to someone else?
Answer: YES or NO
Why: (one specific sentence — what earns it or kills it)

QUESTION 2 — DOES IT HAVE AN OPINION?
Does the writer have a clear point of view, or are they just describing things?
Is there a moment where the writer says something they actually believe, not just reports what others say?
Answer: YES or NO
Which line shows it (or is missing): (quote the line or describe what's missing)

QUESTION 3 — ONE QUOTABLE SENTENCE
Find the single best sentence in the entire blog — the one a reader would screenshot, copy, or remember.
If none exists, say "NONE FOUND" and name which section most needs one.
Best sentence: (quote it or "NONE FOUND — fix [section name]")

QUESTION 4 — DID THE OPENING EARN ATTENTION?
Read the first 3 sentences. Would a smart reader who skims headlines stop and keep reading?
Answer: YES or NO
Issue: (what specifically is wrong, or "None")
Fix: (rewrite the opening in 1–2 sharp sentences, or "None needed")

QUESTION 5 — DOES THE ENDING LAND?
Does the last paragraph say something NEW, or does it just repeat what the blog already said?
Answer: LANDS or RECYCLES
Issue: (what it says now vs what it should say)

OVERALL
Pass: YES or NO (YES only if Q1=YES and at least 3 of Q2-Q5 pass)
Priority Fix: (the single change that would most improve this blog — be specific)

Blog:
{blog[:5_000]}
"""
    result = _invoke(prompt, temperature=0.2)
    logger.info(f"[CRITIC] Done | duration={time.time()-start:.1f}s")
    return result


@traceable(name="targeted_revision")
def targeted_revision(blog: str, critique: str, topic: str, audience: str) -> str:
    """
    Targeted Revision — fixes ONLY the issues flagged by the Human Rubric Critic.
    Surgical and fast — does not rewrite the whole blog.
    """
    _check_cancelled()
    logger.info(f"[REVISION] Starting targeted revision | topic='{topic[:60]}'")
    start = time.time()

    prompt = f"""
You are a precise editor. Fix only what the critic flagged — nothing else.

Topic: {topic}
Audience: {audience}

HUMAN RUBRIC CRITIQUE:
{critique}

WHAT TO FIX:
- If the opening failed Q4 and a rewrite was given — use it
- If no quotable sentence exists (Q3 = NONE FOUND) — add one sharp sentence to the section named
- If the ending recycles (Q5 = RECYCLES) — write a new final paragraph that says something new
- If the blog has no opinion (Q2 = NO) — add one clear point of view in the section that most needs it
- Address the Priority Fix if it was not covered above

RULES:
- Touch ONLY what the critique flagged
- Do NOT rewrite sections that passed
- Do NOT change facts, numbers or citations
- Keep the same voice and style
- Section headings stay as ### prefix
- Return the full revised blog — no commentary

Original Blog:
{blog}
"""
    result = _invoke(prompt, temperature=0.5)
    logger.info(f"[REVISION] Done | duration={time.time()-start:.1f}s | words={len(result.split())}")
    return result
def score_blog(blog: str, topic: str, seo: dict) -> dict:
    prompt = f"""
You are scoring a blog the way a real editor and reader would — not a checklist, actually reading it.

Topic: {topic}

Score each from 1 to 10. Be honest, not generous. A 7 means genuinely good. A 9 means you'd be proud to publish it. A 5 means it needs real work.

CRITERION 1 — DOES IT FEEL HUMAN?
Does it read like a person with a point of view, or like content generated to fill space?
Score: X/10
Reason: (one specific sentence)

CRITERION 2 — OPENING STRENGTH
First 3 sentences only. Would a smart reader stop scrolling?
Not "is it grammatically correct" — does it earn attention?
Score: X/10
Reason: (quote the opening or name what kills it)

CRITERION 3 — ONE ORIGINAL IDEA
Is there a single insight in this blog that the reader couldn't have got from Wikipedia or the first Google result?
Something that makes them think "I never thought about it that way"?
Score: X/10
Reason: (quote the insight or say it's missing)

CRITERION 4 — ENDING QUALITY
Does the last paragraph say something NEW, or summarise what was already said?
Does the reader feel something when they finish?
Score: X/10
Reason: (quote the ending or describe the problem)

CRITERION 5 — WOULD YOU SHARE IT?
If a friend who cares about this topic sent you this blog, would you forward it to someone else?
Score: X/10 (10 = definitely yes, 5 = maybe, 1 = no)
Reason: (one honest sentence)

Overall Score: (average of the 5, one decimal place)
Verdict: (one sharp sentence — what this blog does well and the single thing that would push it higher)

Blog:
{blog[:2_000]}
...
{blog[-1_500:]}

SEO Metadata:
{seo}

Return each criterion on its own line with label followed by colon. Nothing else.
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
