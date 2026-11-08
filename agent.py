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


@traceable(name="plan_blog")
def plan_blog(topic: str, audience: str) -> str:
    _check_cancelled()
    logger.info(f"[PLAN] Starting | topic='{topic[:60]}' | audience='{audience[:40]}'")
    start = time.time()
    result = _invoke(f"""
You are a blog strategist and content analyst.

Topic: {topic}
Target Audience: {audience}

Do two things in one response:

OUTLINE:
1. Title — specific, curiosity-driven, honest.
2. Opening hook — specific tension, surprising fact, or uncomfortable truth.
3. 4 to 6 section headings — each with a one-line description.
4. Core argument — the single most important takeaway.
5. Conclusion angle — challenge, reframe, call to action, or hard truth.
Recommended Length: one word — short / medium / long

COMPETITOR GAP:
In 2-3 sentences: what angle do most articles on this topic take, and what unique angle would make this blog stand out?
""", temperature=0.3)
    logger.info(f"[PLAN] Done | duration={time.time()-start:.1f}s")
    return result

#plan_blog traceble - this decorator sends the functions inpu/ouptut to LangSmith automatically Calls _invoke with a prompt asking for a blog outline + competitor gap analysis . Return a string with the plan .




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


@traceable(name="write_blog")
def write_blog(topic: str, audience: str, plan: str, research_data: str, memory: str, length: str = "medium", gap: str = "", facts: str | None = None) -> str:
    _check_cancelled()
    logger.info(f"[WRITE] Starting | topic='{topic[:60]}' | length={length}")
    start = time.time()
    verified_facts = facts if facts is not None else research_data
    length_instruction = LENGTH_GUIDE.get(length, LENGTH_GUIDE["medium"])
    gap_section = f"\nContent Gap (what existing articles miss — your blog must address this):\n{gap}\n" if gap else ""

    prompt = f"""
You are a writer who has contributed to The Atlantic, Wired, Harvard Business Review, and Vox. You write for humans, not algorithms.

Topic: {topic}
Target Audience: {audience}
Length: {length_instruction}

Blog Plan:
{plan}
{gap_section}

════════════════════════════════════════
RESEARCH DOSSIER — YOUR ONLY FACT SOURCE
════════════════════════════════════════
{verified_facts}
════════════════════════════════════════

Past Context (from memory):
{memory}

══════════════════════════════════════════════════
ABSOLUTE RULES — FACTS (violating these is a failure)
══════════════════════════════════════════════════
1. Every number, percentage, statistic, or named measurement in your blog MUST appear word-for-word in the Research Dossier above.
2. If a number is NOT in the dossier — do NOT write it. Not even a plausible-sounding estimate.
3. Do not use your training knowledge for any factual claim. The dossier is the only truth.
4. CONFIRMED FACTS in the dossier: state them directly and confidently.
5. SINGLE-SOURCE FACTS: attribute them — "according to [source name]..."
6. DISPUTED CLAIMS: write "some analysts suggest X, while others report Y"
7. If the dossier has no numbers for a section — write the insight without numbers. Strong analysis does not require invented statistics.
8. Never write a specific year, match, or event unless it is explicitly named in the dossier.
══════════════════════════════════════════════════

Writing rules — non-negotiable:

Opening:
- First sentence must be specific. Use a real stat from the dossier, a contradiction, a scene, or a hard truth.
- No "In today's world..." or "Have you ever wondered..." ever.
- Talk directly to the reader's exact situation.

Body:
- Every section makes one distinct point. No filler sections.
- After every stat or fact, explain what it means for the reader.
- At least one strong analogy that makes something complex feel simple.
- Short paragraphs — 2 to 4 lines. White space matters.
- Section headings: use ### before each heading. Nothing else — no ##, no bold, no emojis.
- Natural transitions — each section flows into the next.

Tone:
- Match tone to topic and audience.
- Direct. Cut anything that doesn't move the reader forward.
- No corporate speak: "leverage", "synergy", "revolutionize" — banned.
- Smart friend explaining something, not a content marketer.

Conclusion:
- One sharp, specific statement the reader hasn't heard before.
- No "In conclusion..." No recycling the intro.
- No clichés. No question endings. End with a statement that lands.

No hashtags or emojis. Blank line between sections.

Write the full blog now. Remember: every number must come from the dossier.
"""
    result = _invoke(prompt, temperature=0.6)
    logger.info(f"[WRITE] Done | duration={time.time()-start:.1f}s | words={len(result.split())}")
    return result


@traceable(name="audit_facts")
def audit_facts(blog: str, dossier: str, topic: str) -> str:
    """
    Post-write fact audit — finds every number/statistic in the blog,
    checks if it exists in the dossier, removes or replaces hallucinated ones.
    Runs between write and critique_rewrite.
    """
    _check_cancelled()
    logger.info(f"[AUDIT] Starting fact audit | topic='{topic[:60]}'")
    start = time.time()

    prompt = f"""
You are a fact auditor. Your job is to find and fix hallucinated statistics in a blog.

Topic: {topic}

VERIFIED DOSSIER (the only allowed source of facts):
{dossier}

BLOG TO AUDIT:
{blog}

Your task:
1. Find every specific number, percentage, statistic, measurement, or named data point in the blog.
2. For each one — check if it exists in the dossier above.
3. If it EXISTS in the dossier → keep it exactly as is.
4. If it does NOT exist in the dossier → do one of these:
   a. If the dossier has the correct version of this fact → replace with the correct dossier value
   b. If the dossier has no version of this fact → remove the number and rephrase without it
      Example: "his strike rate of 90 in Tests" → "his strike rate in Tests" or remove the sentence
5. Do NOT add any new facts that aren't in the dossier.
6. Do NOT change anything else — tone, structure, headings, paragraphs must stay identical.
7. If a sentence becomes awkward after removing a number, rewrite only that sentence naturally.

Return the full corrected blog. No commentary, no explanation — just the blog.
"""
    result = _invoke(prompt, temperature=0.0)
    logger.info(f"[AUDIT] Done | duration={time.time()-start:.1f}s")
    return result


@traceable(name="critique_and_rewrite")
def critique_and_rewrite(blog: str, topic: str, audience: str) -> str:
    _check_cancelled()
    logger.info(f"[CRITIQUE] Starting rewrite | topic='{topic[:60]}'")
    start = time.time()
    prompt = f"""
You are a senior editor from Wired and The Atlantic. Ruthless about quality.

Topic: {topic}
Target Audience: {audience}

Rewrite this blog as the final published version. Fix:
- Weak or generic opening — replace with something that grabs immediately
- Any stat dropped without explanation — add the "so what"
- Any cliché — cut it, replace with something specific
- Any filler section — make it earn its place or cut it
- Weak conclusion — one sharp, memorable final thought
- Choppy transitions — smooth them

Check:
- At least one strong analogy?
- Every paragraph moves the reader forward?
- Conclusion leaves the reader with something new?

Rules:
- No emojis
- Section headings must use ### prefix — nothing else
- Short paragraphs, blank line between sections
- No clichés: "journey not a destination", "unlock potential", "game changer" — banned
- Write like a smart human, not a content farm

Return only the final blog. No commentary.

Original Blog:
{blog}
"""
    result = _invoke(prompt, temperature=0.6)
    logger.info(f"[CRITIQUE] Done | duration={time.time()-start:.1f}s | words={len(result.split())}")
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
    """
    import re

    # Parse the structured facts list
    source_map = {}  # citation_num -> (claim, title, url)
    for line in facts.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Format: [N] claim | source_title | source_url
        m = re.match(r'^\[(\d+)\]\s+(.+?)\s+\|\s+(.+?)\s+\|\s+(https?://\S+)', line)
        if m:
            num, claim, title, url = m.group(1), m.group(2), m.group(3), m.group(4)
            source_map[num] = (claim.strip(), title.strip(), url.strip())

    if not source_map:
        # No structured facts available — return blog as-is with no Sources
        logger.warning("[CITATIONS] No structured facts found — skipping citations")
        return blog

    # Add inline citation markers to the blog
    # For each fact, find its key words in the blog and append [N] after the sentence
    cited_blog = blog
    used_citations = []

    for num, (claim, title, url) in source_map.items():
        # Extract the most distinctive 4-6 word phrase from the claim
        claim_words = [w for w in claim.lower().split() if len(w) > 3][:5]
        if not claim_words:
            continue

        # Search for a sentence containing these words
        pattern = r'([^.!?]*' + re.escape(claim_words[0]) + r'[^.!?]*[.!?])'
        matches = list(re.finditer(pattern, cited_blog, re.IGNORECASE))

        for match in matches:
            sentence = match.group(0)
            # Check if enough claim words appear in this sentence
            words_found = sum(1 for w in claim_words if w in sentence.lower())
            if words_found >= min(2, len(claim_words)):
                # Only add citation if not already cited at this spot
                marker = f"[{num}]"
                if marker not in cited_blog[max(0, match.end()-20):match.end()+5]:
                    # Insert citation before the sentence-ending punctuation
                    cited_blog = cited_blog[:match.end()-1] + marker + cited_blog[match.end()-1:]
                    used_citations.append(num)
                    break  # only cite once per claim

    # Build Sources section from actually used citations
    # Fall back to all sources if no inline citations were placed
    citations_to_show = used_citations if used_citations else list(source_map.keys())
    # Deduplicate while preserving order
    seen = set()
    citations_to_show = [n for n in citations_to_show if not (n in seen or seen.add(n))]

    sources_lines = ["Sources"]
    for num in citations_to_show:
        if num in source_map:
            _, title, url = source_map[num]
            sources_lines.append(f"[{num}] {title} — {url}")

    # Also add any remaining sources that weren't inline-cited
    for num, (_, title, url) in source_map.items():
        if num not in seen:
            sources_lines.append(f"[{num}] {title} — {url}")

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
