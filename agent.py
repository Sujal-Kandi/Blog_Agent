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



def analyze_competitor_gap(topic: str) -> str:
    # Merged into plan_blog to save LLM calls
    return ""
    queries = [
        topic,
        f"{topic} surprising facts and lesser known insights",
        f"{topic} recent data statistics research 2024 2025",
        f"{topic} common misconceptions and expert perspectives"
    ]

    def fetch(q):
        result = web_search(q)
        return f"Query: {q}\n{result}"

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(fetch, queries))

    return "\n\n".join(results)


@traceable(name="research")
def research(topic: str) -> str:
    _check_cancelled()
    logger.info(f"[RESEARCH] Starting web search | topic='{topic[:60]}'")
    start = time.time()
    queries = [
        topic,
        f"{topic} surprising facts and lesser known insights",
        f"{topic} recent data statistics research 2024 2025",
        f"{topic} common misconceptions and expert perspectives"
    ]

    def fetch(q):
        _check_cancelled()  # skip fetch entirely if already cancelled
        result = web_search(q)
        return f"Query: {q}\n{result}"

    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fetch, q): q for q in queries}
        for future in as_completed(futures):
            if _cancel_event.is_set():
                # Cancel remaining futures and exit early
                for f in futures:
                    f.cancel()
                raise RuntimeError("Pipeline cancelled during research.")
            try:
                results.append(future.result())
            except RuntimeError:
                raise  # propagate cancellation
            except Exception as e:
                logger.warning(f"[RESEARCH] fetch failed: {e}")
                results.append(f"Query: {futures[future]}\n[fetch failed: {e}]")

    combined = "\n\n".join(results)
    logger.info(f"[RESEARCH] Done | duration={time.time()-start:.1f}s | chars={len(combined)}")
    return combined


@traceable(name="extract_facts")
def extract_facts(topic: str, research_data: str) -> str:
    _check_cancelled()
    prompt = f"""
You are a research analyst. Extract only the most concrete, specific, and useful information.

Topic: {topic}

From the research below, extract:
- Specific statistics with their source in brackets e.g. "67% of developers use Copilot daily [GitHub Survey 2024]"
- Named tools, companies, or products that are relevant
- Real examples of what is actually happening right now
- Surprising or counterintuitive findings
- Direct quotes from named experts or studies with attribution e.g. "Quote here" — Name, Title
- Anything most people don't know about this topic

Every fact MUST have a source in brackets. If no source is available, skip it.
Return a clean numbered list of concrete facts only.

Research Data:
{research_data}
"""
    return _invoke(prompt, temperature=0.3)


@traceable(name="write_blog")
def write_blog(topic: str, audience: str, plan: str, research_data: str, memory: str, length: str = "medium", gap: str = "", facts: str | None = None) -> str:
    _check_cancelled()
    logger.info(f"[WRITE] Starting | topic='{topic[:60]}' | length={length}")
    start = time.time()
    verified_facts = facts if facts is not None else extract_facts(topic, research_data)
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
Verified Facts (use these — do not invent statistics):
{verified_facts}

Past Context (from memory):
{memory}

Rules — non-negotiable:

Opening:
- First sentence must be specific. A stat, a contradiction, a scene, or an uncomfortable truth.
- No "In today's world..." or "Have you ever wondered..." ever.
- Talk directly to the reader's exact situation.

Body:
- Every section makes one distinct point. No filler sections.
- After every stat or fact, explain what it means for the reader.
- At least one strong analogy that makes something complex feel simple.
- Short paragraphs — 2 to 4 lines. White space matters.
- Plain text section headings. No hashtags, no bold markers, no emojis.
- Depth is non-negotiable at any length. Short does not mean shallow.
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

Write the full blog now.
"""
    result = _invoke(prompt, temperature=0.6)
    logger.info(f"[WRITE] Done | duration={time.time()-start:.1f}s | words={len(result.split())}")
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
- No hashtags or emojis
- Plain text headings
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
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("- "):
            extras["tldr"].append(line[2:].strip())
        elif line.lower().startswith("pull quote:"):
            extras["pull_quote"] = line.partition(":")[2].strip()
        elif line.lower().startswith("key takeaway:"):
            extras["key_takeaway"] = line.partition(":")[2].strip()

    return extras


def generate_citations(blog: str, facts: str, topic: str) -> str:
    prompt = f"""
You are a fact-checker and citations editor.

Topic: {topic}

Below is a blog and the verified facts with sources that were used to write it.

Your job:
1. Find every statistic, data point, or named claim in the blog that matches a fact from the list
2. Add an inline citation marker after it like [1], [2] etc.
3. At the end of the blog, add a "Sources" section listing each citation

Rules:
- Only cite facts that actually appear in the blog
- Keep the citation markers minimal and unobtrusive
- The Sources section should be clean: [1] Source name or URL
- Do not change any other part of the blog

Verified Facts with Sources:
{facts}

Blog:
{blog}

Return the full blog with inline citations and Sources section appended at the end.
"""
    return _invoke(prompt, temperature=0.0)


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
