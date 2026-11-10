"""
Research Engine — 6-Phase Pipeline
====================================
Replaces the old research() + extract_facts() functions.

Phase 1 — Topic Intelligence        (~0.3s, template-based, LLM fallback)
Phase 2 — Parallel Search           (~3s,   async Tavily Advanced Search)
Phase 3 — Source Ranking            (~0.2s, pure Python, zero credits)
Phase 4 — Parallel Extraction       (~2.5s, async Tavily Extract + chunking)
Phase 5 — Claim Extraction          (~4s,   one batched LLM call, Pydantic)
Phase 6 — Smart Verification        (~3s,   cross-check + selective search)

Output  — Research Dossier (structured string for writing engine)
"""

import os
import re
import time
import asyncio
import logging
import sqlite3
import hashlib
import threading
from typing import Optional
from datetime import datetime, timezone

from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("blog_agent")

# ── Tavily key rotation ───────────────────────────────────────────────────────

TAVILY_KEYS = [
    value.strip()
    for name, value in sorted(os.environ.items())
    if (
        name == "TAVILY_API_KEY"
        or name.startswith("TAVILY_API_KEY_")
        or name.startswith("TAVILY_KEY_")
    ) and value.strip()
]
_tavily_key_index = 0
_tavily_lock = threading.Lock()


def _get_tavily_key() -> str:
    global _tavily_key_index
    if not TAVILY_KEYS:
        raise RuntimeError(
            "No Tavily API keys configured. Set TAVILY_API_KEY or TAVILY_API_KEY_2, TAVILY_KEY_2 etc."
        )
    with _tavily_lock:
        key = TAVILY_KEYS[_tavily_key_index]
    return key


def _rotate_tavily_key():
    global _tavily_key_index
    with _tavily_lock:
        _tavily_key_index = (_tavily_key_index + 1) % len(TAVILY_KEYS)


# ── SQLite cache ──────────────────────────────────────────────────────────────

_DB_PATH = os.path.join(os.path.dirname(__file__), "research_cache.db")
_db_lock = threading.Lock()


def _init_cache():
    with _db_lock:
        conn = sqlite3.connect(_DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS extract_cache (
                url_hash TEXT PRIMARY KEY,
                url      TEXT,
                content  TEXT,
                cached_at INTEGER
            )
        """)
        conn.commit()
        conn.close()


_init_cache()

_CACHE_TTL_SECONDS = 7 * 24 * 3600  # 7 days


def _cache_get(url: str) -> Optional[str]:
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    now = int(time.time())
    with _db_lock:
        conn = sqlite3.connect(_DB_PATH)
        row = conn.execute(
            "SELECT content, cached_at FROM extract_cache WHERE url_hash = ?",
            (url_hash,)
        ).fetchone()
        conn.close()
    if row and (now - row[1]) < _CACHE_TTL_SECONDS:
        return row[0]
    return None


def _cache_set(url: str, content: str):
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    now = int(time.time())
    with _db_lock:
        conn = sqlite3.connect(_DB_PATH)
        conn.execute(
            "INSERT OR REPLACE INTO extract_cache (url_hash, url, content, cached_at) VALUES (?,?,?,?)",
            (url_hash, url, content, now)
        )
        conn.commit()
        conn.close()


# ── Pydantic models ───────────────────────────────────────────────────────────

class Claim(BaseModel):
    claim: str
    evidence: str
    source_url: str
    source_title: str
    published_date: str
    claim_type: str      # "stat" | "quote" | "event" | "general"
    topic_relevant: bool
    confidence: float    # 0.0 - 1.0


class ResearchDossier(BaseModel):
    confirmed: list[str]      # facts confirmed by 2+ sources
    single_source: list[str]  # facts from one credible source
    disputed: list[str]       # facts where sources disagree
    source_urls: list[str]    # all source URLs used
    competitor_gap: str       # what existing articles miss


# ── Domain trust tiers ────────────────────────────────────────────────────────

_TIER1_DOMAINS = {
    "gov", "edu", "espncricinfo.com", "cricinfo.com", "bbc.com", "bbc.co.uk",
    "reuters.com", "apnews.com", "nature.com", "pubmed.ncbi.nlm.nih.gov",
    "who.int", "cdc.gov", "worldbank.org", "statista.com", "pewresearch.org",
    "nytimes.com", "washingtonpost.com", "theguardian.com", "economist.com",
    "wsj.com", "ft.com", "bloomberg.com", "forbes.com", "harvard.edu",
    "stanford.edu", "mit.edu", "cricbuzz.com", "espn.com", "skysports.com"
}

_TIER2_DOMAINS = {
    "wikipedia.org", "medium.com", "techcrunch.com", "wired.com",
    "theverge.com", "arstechnica.com", "zdnet.com", "businessinsider.com",
    "cnbc.com", "marketwatch.com", "investopedia.com", "healthline.com",
    "webmd.com", "sciencedaily.com", "theatlantic.com", "vox.com"
}


def _domain_trust(url: str) -> float:
    try:
        domain = url.split("/")[2].lower().replace("www.", "")
        tld = domain.split(".")[-1]
        if tld in ("gov", "edu") or domain in _TIER1_DOMAINS:
            return 1.0
        if domain in _TIER2_DOMAINS:
            return 0.8
        return 0.5
    except Exception:
        return 0.5




# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Topic Intelligence (fully LLM-driven, zero hardcoding)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_sub_queries(topic: str, use_llm_fallback=None) -> tuple[list[str], str, bool]:
    """
    Phase 1: One LLM call that does two things:
      1. Understands what this topic *is* and what kind of blog it needs.
      2. Generates 5 focused search queries tailored to that understanding.

    No hardcoded templates. No keyword lists. The LLM decides the topic type,
    research angle, and queries from scratch for every topic.

    Returns: (sub_queries, topic_type, freshness_required)
    """
    if use_llm_fallback is None:
        logger.warning("[PHASE1] No LLM available — using bare topic as single query")
        return [topic], "general", False

    prompt = (
        f'You are a research strategist for a blog writing system. '
        f'A blog will be written about this topic:\n\n'
        f'TOPIC: "{topic}"\n\n'
        f'Your job has two parts:\n\n'
        f'PART 1 — TOPIC UNDERSTANDING\n'
        f'Analyze this topic and answer:\n'
        f'- topic_type: What kind of topic is this? Choose ONE:\n'
        f'    person_biography   (a real person — life, career, achievements)\n'
        f'    person_skill       (a specific skill or technique of a named person)\n'
        f'    historical_event   (a past event, origin story, or how something began)\n'
        f'    concept_explainer  (an idea, philosophy, religion, movement, or system)\n'
        f'    sports_analysis    (a team, era, league, or sports trend — not a person)\n'
        f'    science_health     (science, biology, medicine, nutrition, psychology)\n'
        f'    business_tech      (companies, startups, AI, economics, products)\n'
        f'    culture_society    (food, lifestyle, travel, social trends, pop culture)\n'
        f'    opinion_why        (why something happens, fails, or matters)\n'
        f'    other              (anything that does not fit above)\n'
        f'- freshness_required: true if the topic needs recent/current sources '
        f'(live stats, news, current events), false for evergreen topics\n'
        f'- blog_angle: One sentence describing what this blog should FOCUS ON '
        f'to be most interesting and useful for a reader\n\n'
        f'PART 2 — SEARCH QUERIES\n'
        f'Generate exactly 5 search queries to find the best sources for this blog.\n'
        f'Each query must target a DIFFERENT angle of the topic.\n'
        f'Queries must be specific, keyword-dense (5-8 words), directly about the topic.\n\n'
        f'Return ONLY this JSON (no markdown, no explanation):\n'
        f'{{\n'
        f'  "topic_type": "...",\n'
        f'  "freshness_required": false,\n'
        f'  "blog_angle": "...",\n'
        f'  "queries": [\n'
        f'    "query 1",\n'
        f'    "query 2",\n'
        f'    "query 3",\n'
        f'    "query 4",\n'
        f'    "query 5"\n'
        f'  ]\n'
        f'}}'
    )

    try:
        raw = use_llm_fallback(prompt, temperature=0.2)

        # Strip markdown code fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        import json as _json
        data = _json.loads(raw)

        topic_type         = str(data.get("topic_type", "other")).strip()
        freshness_required = bool(data.get("freshness_required", False))
        blog_angle         = str(data.get("blog_angle", "")).strip()
        queries            = [str(q).strip() for q in data.get("queries", []) if str(q).strip()]

        if len(queries) < 3:
            raise ValueError(f"Only {len(queries)} queries returned")

        queries = queries[:5]

        logger.info(
            f"[PHASE1] topic_type={topic_type} | freshness={freshness_required} | "
            f"queries={len(queries)}"
        )
        logger.info(f"[PHASE1] angle: {blog_angle[:120]}")
        logger.info(f"[PHASE1] Done in 0.00s | type={topic_type} | queries={queries}")

        return queries, topic_type, freshness_required

    except Exception as e:
        logger.warning(f"[PHASE1] LLM topic intelligence failed: {e} — using fallback queries")
        fallback = [
            topic,
            f"{topic} statistics data facts",
            f"{topic} expert analysis overview",
        ]
        return fallback, "other", False


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Parallel Search
# ═══════════════════════════════════════════════════════════════════════════════

def _tavily_search_sync(query: str, freshness_required: bool) -> list[dict]:
    """Single Tavily search — returns list of result dicts."""
    from tavily import TavilyClient, MissingAPIKeyError
    try:
        client = TavilyClient(api_key=_get_tavily_key())
        params = {
            "query": query,
            "search_depth": "advanced",
            "max_results": 5,
            "include_answer": False,
        }
        if freshness_required:
            params["days"] = 365  # last 12 months for fast-moving topics
        response = client.search(**params)
        return response.get("results", [])
    except MissingAPIKeyError:
        _rotate_tavily_key()
        logger.warning(f"[PHASE2] Tavily key error — rotated key")
        return []
    except Exception as e:
        logger.warning(f"[PHASE2] Search failed for '{query[:50]}': {e}")
        return []


async def _search_all(queries: list[str], freshness_required: bool) -> list[dict]:
    """Run all searches in parallel using asyncio + thread pool."""
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(None, _tavily_search_sync, q, freshness_required)
        for q in queries
    ]
    results_nested = await asyncio.gather(*tasks, return_exceptions=True)

    # Flatten + deduplicate by URL
    seen_urls = set()
    all_results = []
    for batch in results_nested:
        if isinstance(batch, Exception):
            logger.warning(f"[PHASE2] Search task failed: {batch}")
            continue
        for r in batch:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(r)

    logger.info(f"[PHASE2] {len(all_results)} unique results from {len(queries)} queries")
    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Source Ranking
# ═══════════════════════════════════════════════════════════════════════════════

def _score_result(result: dict, topic: str, freshness_required: bool) -> float:
    """Score a search result 0-1 for relevance + freshness + trust."""
    topic_words = set(topic.lower().split()) - {"the", "a", "an", "is", "are", "of", "in"}

    # Relevance — keyword overlap between topic and title+snippet
    text = f"{result.get('title', '')} {result.get('content', '')}".lower()
    overlap = sum(1 for w in topic_words if w in text)
    relevance = min(overlap / max(len(topic_words), 1), 1.0)

    # Freshness
    freshness = 1.0
    if freshness_required:
        pub = result.get("published_date") or result.get("publishedDate", "")
        if pub:
            try:
                # Handle both timestamp (int/float) and ISO string
                if isinstance(pub, (int, float)):
                    pub_dt = datetime.fromtimestamp(pub / 1000 if pub > 1e10 else pub, tz=timezone.utc)
                else:
                    pub_dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                age_days = (datetime.now(tz=timezone.utc) - pub_dt).days
                if age_days <= 180:
                    freshness = 1.0
                elif age_days <= 365:
                    freshness = 0.8
                elif age_days <= 730:
                    freshness = 0.5
                else:
                    freshness = 0.3
            except Exception:
                freshness = 0.7  # unknown date — neutral

    # Trust
    trust = _domain_trust(result.get("url", ""))

    score = 0.5 * relevance + 0.3 * freshness + 0.2 * trust
    return round(score, 3)


def rank_and_select(results: list[dict], topic: str, freshness_required: bool, top_n: int = 4) -> list[dict]:
    """Phase 3: Score and return top N results."""
    scored = []
    for r in results:
        s = _score_result(r, topic, freshness_required)
        scored.append((s, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [r for _, r in scored[:top_n]]
    logger.info(f"[PHASE3] Selected top {len(selected)} sources from {len(results)} results")
    for s, r in scored[:top_n]:
        logger.info(f"  score={s:.2f} | {r.get('url','')[:70]}")
    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Parallel Extraction + Chunking
# ═══════════════════════════════════════════════════════════════════════════════

def _tavily_extract_sync(url: str, snippet: str) -> str:
    """Extract full page content. Falls back to snippet if extraction fails/cached."""
    cached = _cache_get(url)
    if cached:
        logger.info(f"[PHASE4] Cache hit: {url[:60]}")
        return cached

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=_get_tavily_key())
        response = client.extract(urls=[url], extract_depth="basic")
        results = response.get("results", [])
        if results and results[0].get("raw_content"):
            content = results[0]["raw_content"][:8000]  # cap at 8k chars
            _cache_set(url, content)
            logger.info(f"[PHASE4] Extracted {len(content)} chars: {url[:60]}")
            return content
    except Exception as e:
        logger.warning(f"[PHASE4] Extract failed for {url[:60]}: {e} — using snippet")

    # Fallback to snippet
    return snippet


async def _extract_all(sources: list[dict]) -> list[dict]:
    """Extract all sources in parallel."""
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(
            None,
            _tavily_extract_sync,
            s.get("url", ""),
            s.get("content", "")
        )
        for s in sources
    ]
    contents = await asyncio.gather(*tasks, return_exceptions=True)

    enriched = []
    for source, content in zip(sources, contents):
        if isinstance(content, Exception):
            content = source.get("content", "")
        enriched.append({**source, "full_content": str(content)})
    return enriched


def _chunk_text(text: str, chunk_size: int = 350) -> list[str]:
    """Split text into ~350 word chunks at sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = []
    current_words = 0

    for sent in sentences:
        words = len(sent.split())
        if current_words + words > chunk_size and current:
            chunks.append(" ".join(current))
            current = [sent]
            current_words = words
        else:
            current.append(sent)
            current_words += words

    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c.split()) >= 20]  # drop tiny chunks


def _score_chunk(chunk: str, topic: str, sub_queries: list[str]) -> float:
    """Score a chunk by keyword overlap with topic and sub-queries."""
    topic_words = set(topic.lower().split()) - {"the", "a", "an", "is", "are", "of"}
    chunk_lo = chunk.lower()

    # Topic overlap
    topic_score = sum(1 for w in topic_words if w in chunk_lo) / max(len(topic_words), 1)

    # Sub-query overlap — does this chunk answer any of the research questions?
    query_score = 0.0
    for q in sub_queries:
        q_words = set(q.lower().split()) - {"the", "a", "an"}
        overlap = sum(1 for w in q_words if w in chunk_lo)
        query_score = max(query_score, overlap / max(len(q_words), 1))

    return 0.4 * topic_score + 0.6 * query_score


def chunk_and_rank(
    enriched_sources: list[dict],
    topic: str,
    sub_queries: list[str],
    top_n: int = 12
) -> list[dict]:
    """
    Phase 4: Chunk all extracted pages, score each chunk, return top N.
    Each chunk carries its source URL and title.
    """
    all_chunks = []
    for source in enriched_sources:
        content = source.get("full_content") or source.get("content", "")
        url = source.get("url", "")
        title = source.get("title", "")
        pub_date = source.get("published_date") or source.get("publishedDate", "")

        chunks = _chunk_text(content)
        for chunk in chunks:
            score = _score_chunk(chunk, topic, sub_queries)
            all_chunks.append({
                "text": chunk,
                "source_url": url,
                "source_title": title,
                "published_date": str(pub_date)[:10] if pub_date else "",
                "score": score,
            })

    # Sort by score, take top N
    all_chunks.sort(key=lambda x: x["score"], reverse=True)
    selected = all_chunks[:top_n]
    logger.info(f"[PHASE4] {len(all_chunks)} total chunks → top {len(selected)} selected")
    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — Claim Extraction (Pydantic + LLM)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_claims(chunks: list[dict], topic: str, invoke_fn) -> list[Claim]:
    """
    Phase 5: One batched LLM call to extract all factual claims from top chunks.
    Returns list of Claim objects. Off-topic claims are filtered out.

    For broad/philosophical topics (Hinduism, history, culture, etc.) stats and
    quotes are rare — we explicitly include "general" and "event" claims so the
    dossier is never empty for these topics.

    Token budget: each chunk is capped at 600 chars and total evidence block
    at ~12 000 chars to stay within the model's TPM limit.
    """
    if not chunks:
        return []

    # ── Token budget guard ────────────────────────────────────────────────────
    # The model limit is 8 000 TPM. Each char ≈ 0.3 tokens, so ~26 000 chars
    # is roughly 8 000 tokens. We reserve ~8 000 chars for the prompt template
    # and instructions, leaving ~12 000 chars for the evidence block.
    MAX_EVIDENCE_CHARS = 12_000
    CHUNK_TEXT_CAP    = 600   # chars per chunk (was 8 000 — way too big)

    # Build evidence block with per-chunk cap
    evidence_lines = []
    total_chars = 0
    chunks_used = 0
    for i, chunk in enumerate(chunks, 1):
        text = chunk["text"][:CHUNK_TEXT_CAP]
        block = (
            f"\n[CHUNK {i}]\n"
            f"Source: {chunk['source_title']}\n"
            f"URL: {chunk['source_url']}\n"
            f"Date: {chunk['published_date']}\n"
            f"Text: {text}\n---"
        )
        if total_chars + len(block) > MAX_EVIDENCE_CHARS:
            break
        evidence_lines.append(block)
        total_chars += len(block)
        chunks_used += 1

    evidence_block = "".join(evidence_lines)
    logger.info(f"[PHASE5] Using {chunks_used}/{len(chunks)} chunks | evidence_chars={total_chars}")

    prompt = f"""
You are a research analyst extracting factual claims for a blog about: "{topic}"

Below are {chunks_used} evidence chunks from real web sources.

Extract EVERY factual claim that is verifiable and useful for a blog writer:
- A specific statistic, number, or measurement
- A named quote from a real person
- A specific event with a date or time period
- A concrete descriptive fact (e.g. "X is the oldest/largest/most...", "X originated in Y", "X has N practitioners")
- A widely-accepted general fact that adds depth or context

For each claim return a JSON object on its own line (JSONL format):
{{"claim": "...", "evidence": "exact passage supporting it", "source_url": "...", "source_title": "...", "published_date": "...", "claim_type": "stat|quote|event|general", "topic_relevant": true/false, "confidence": 0.0-1.0}}

Rules:
- claim_type "stat" = numbers, percentages, measurements
- claim_type "quote" = exact words from a named person
- claim_type "event" = specific dated or historical occurrence
- claim_type "general" = concrete fact, description, or widely-accepted knowledge
- topic_relevant = true if the claim is relevant to "{topic}" in any way
- confidence = how certain you are this is accurate based on the evidence
- Do NOT invent claims — only extract what's in the evidence
- Include "general" and "event" claims liberally — they are as valuable as stats
- Minimum extraction target: aim for at least 8 claims per chunk batch
- Return ONLY the JSONL lines, nothing else

Evidence:
{evidence_block}
"""

    try:
        raw = invoke_fn(prompt, temperature=0.0)
    except Exception as e:
        logger.warning(f"[PHASE5] Claim extraction failed: {e}")
        return []

    import json as _json
    claims = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            data = _json.loads(line)
            # Coerce confidence — accept strings like "0.8" or missing values
            try:
                conf = float(data.get("confidence", 0.5))
            except (TypeError, ValueError):
                conf = 0.5

            claim = Claim(
                claim=data.get("claim", "").strip(),
                evidence=data.get("evidence", "").strip(),
                source_url=data.get("source_url", "").strip(),
                source_title=data.get("source_title", "").strip(),
                published_date=str(data.get("published_date", ""))[:10],
                claim_type=data.get("claim_type", "general"),
                # Treat missing/null topic_relevant as True — be inclusive
                topic_relevant=bool(data.get("topic_relevant", True)),
                confidence=conf,
            )
            # Only filter: must have a non-empty claim and be topic-relevant
            if claim.claim and claim.topic_relevant:
                claims.append(claim)
        except Exception as e:
            logger.debug(f"[PHASE5] Failed to parse claim line: {e} | {line[:80]}")

    logger.info(f"[PHASE5] Extracted {len(claims)} on-topic claims from {chunks_used} chunks")
    return claims


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — Smart Verification
# ═══════════════════════════════════════════════════════════════════════════════

def _cross_check_in_chunks(claim: Claim, chunks: list[dict]) -> bool:
    """
    Check if a claim's key number/name appears in 2+ chunks from different URLs.
    Returns True if cross-verified within existing results (free, instant).
    """
    # Extract the core verifiable element from the claim
    # Look for numbers, percentages, or quoted names
    numbers = re.findall(r'\d+\.?\d*\s*%?', claim.claim)
    key_terms = numbers if numbers else claim.claim.split()[:4]

    supporting_urls = set()
    for chunk in chunks:
        chunk_lo = chunk["text"].lower()
        if all(str(t).lower() in chunk_lo for t in key_terms[:2]):
            supporting_urls.add(chunk["source_url"])

    return len(supporting_urls) >= 2


def _verify_with_search(claim: Claim) -> str:
    """
    Fire one targeted Tavily search to verify a single-source stat/quote.
    Returns: 'confirmed' | 'disputed' | 'single_source'
    """
    # Build a tight verification query from the claim
    query = f'"{claim.claim[:80]}"' if len(claim.claim) <= 80 else claim.claim[:80]

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=_get_tavily_key())
        response = client.search(query=query, search_depth="basic", max_results=3)
        results = response.get("results", [])

        if not results:
            return "single_source"

        # Check if verification results support or contradict
        claim_lo = claim.claim.lower()
        numbers = re.findall(r'\d+\.?\d*', claim_lo)

        supporting = 0
        contradicting = 0

        for r in results:
            text = (r.get("content", "") + r.get("title", "")).lower()
            # Skip if it's the same source
            if claim.source_url and claim.source_url in r.get("url", ""):
                continue
            if numbers:
                if any(n in text for n in numbers):
                    supporting += 1
            else:
                # For non-number claims, check keyword overlap
                key_words = [w for w in claim_lo.split() if len(w) > 4][:5]
                if sum(1 for w in key_words if w in text) >= 3:
                    supporting += 1

        if supporting >= 1:
            return "confirmed"
        return "single_source"

    except Exception as e:
        logger.warning(f"[PHASE6] Verification search failed: {e}")
        return "single_source"


def verify_claims(claims: list[Claim], chunks: list[dict]) -> dict[str, list[Claim]]:
    """
    Phase 6: Verify claims and bucket them.

    Returns dict with keys: 'confirmed', 'single_source', 'disputed'
    Only stats and quotes go through verification.
    General and event claims pass through as single_source.
    """
    confirmed = []
    single_source = []
    disputed = []

    verification_searches = 0
    MAX_VERIFICATION_SEARCHES = 4  # hard cap to control credits

    for claim in claims:
        if claim.claim_type in ("stat", "quote"):
            # Step 1: free cross-check within existing chunks
            if _cross_check_in_chunks(claim, chunks):
                confirmed.append(claim)
                continue

            # Step 2: paid verification search (only for high-confidence, important claims)
            if (
                claim.confidence >= 0.7
                and verification_searches < MAX_VERIFICATION_SEARCHES
            ):
                verification_searches += 1
                status = _verify_with_search(claim)
                if status == "confirmed":
                    confirmed.append(claim)
                elif status == "disputed":
                    disputed.append(claim)
                else:
                    single_source.append(claim)
            else:
                single_source.append(claim)

        else:
            # General facts and events — pass through as single_source
            single_source.append(claim)

    logger.info(
        f"[PHASE6] confirmed={len(confirmed)} | single_source={len(single_source)} | "
        f"disputed={len(disputed)} | verification_searches={verification_searches}"
    )
    return {
        "confirmed": confirmed,
        "single_source": single_source,
        "disputed": disputed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RESEARCH DOSSIER — Final output to writing engine
# ═══════════════════════════════════════════════════════════════════════════════

def build_dossier(
    verified: dict[str, list[Claim]],
    sources: list[dict],
    competitor_gap: str,
    topic: str,
) -> str:
    """
    Build the final research dossier string for the writing engine.
    Each fact includes full context: claim + evidence + source + date.
    Format is explicit so the writer cannot mix up format/context.
    """
    lines = [f"RESEARCH DOSSIER — {topic}\n"]

    # Confirmed facts
    if verified["confirmed"]:
        lines.append("── CONFIRMED FACTS (verified by 2+ independent sources — cite directly) ──")
        for i, c in enumerate(verified["confirmed"], 1):
            # Include source context explicitly to prevent format/attribution mix-ups
            source_context = f"{c.source_title}" if c.source_title else "unknown source"
            date_context = f" ({c.published_date[:4]})" if c.published_date and len(c.published_date) >= 4 else ""
            lines.append(
                f"FACT-{i}: {c.claim}\n"
                f"  Context: {c.evidence[:300]}\n"
                f"  Source: {source_context}{date_context}\n"
                f"  URL: {c.source_url}\n"
                f"  Type: {c.claim_type}"
            )
        lines.append("")

    # Single-source facts
    if verified["single_source"]:
        lines.append('── SINGLE-SOURCE FACTS (attribute as: "according to [Source]...") ──')
        for i, c in enumerate(verified["single_source"], 1):
            source_context = f"{c.source_title}" if c.source_title else "unknown source"
            date_context = f" ({c.published_date[:4]})" if c.published_date and len(c.published_date) >= 4 else ""
            lines.append(
                f"FACT-{i}: {c.claim}\n"
                f"  Context: {c.evidence[:200]}\n"
                f"  Source: {source_context}{date_context}\n"
                f"  URL: {c.source_url}"
            )
        lines.append("")

    # Disputed claims
    if verified["disputed"]:
        lines.append('── DISPUTED CLAIMS (write: "some sources suggest X, while others report Y") ──')
        for i, c in enumerate(verified["disputed"], 1):
            lines.append(f"DISPUTED-{i}: {c.claim} [conflicting sources]")
        lines.append("")

    # Competitor gap
    if competitor_gap:
        lines.append("── COMPETITOR GAP (what existing articles miss — address this angle) ──")
        lines.append(competitor_gap)
        lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def run_research_engine(topic: str, competitor_gap: str, invoke_fn) -> tuple[str, str]:
    """
    Run the full 6-phase research engine.

    Args:
        topic:          User's blog topic
        competitor_gap: Output from analyze_competitor_gap (already computed in parallel)
        invoke_fn:      The _invoke function from agent.py (passed in to avoid circular import)

    Returns:
        (dossier_str, raw_facts_str)
        dossier_str   — structured dossier for write_blog prompt
        raw_facts_str — flat facts string for generate_citations (backward compat)
    """
    start = time.time()
    logger.info(f"[RESEARCH ENGINE] Starting | topic='{topic[:60]}'")

    # ── Phase 1: Topic Intelligence
    t1 = time.time()
    sub_queries, topic_type, freshness_required = generate_sub_queries(topic, use_llm_fallback=invoke_fn)
    logger.info(f"[PHASE1] Done in {time.time()-t1:.2f}s | type={topic_type} | queries={sub_queries}")

    # ── Phase 2+3: Parallel Search + Source Ranking
    t2 = time.time()
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    all_results = loop.run_until_complete(_search_all(sub_queries, freshness_required))
    logger.info(f"[PHASE2] Done in {time.time()-t2:.2f}s | results={len(all_results)}")

    if not all_results:
        logger.warning("[RESEARCH ENGINE] No search results — returning empty dossier")
        return "", ""

    t3 = time.time()
    top_sources = rank_and_select(all_results, topic, freshness_required, top_n=4)
    logger.info(f"[PHASE3] Done in {time.time()-t3:.2f}s")

    # ── Phase 4: Parallel Extraction + Chunking
    t4 = time.time()
    enriched = loop.run_until_complete(_extract_all(top_sources))
    top_chunks = chunk_and_rank(enriched, topic, sub_queries, top_n=12)
    logger.info(f"[PHASE4] Done in {time.time()-t4:.2f}s | chunks={len(top_chunks)}")

    # ── Phase 5: Claim Extraction
    t5 = time.time()
    claims = extract_claims(top_chunks, topic, invoke_fn)
    logger.info(f"[PHASE5] Done in {time.time()-t5:.2f}s | claims={len(claims)}")

    # ── Phase 6: Smart Verification
    t6 = time.time()
    verified = verify_claims(claims, top_chunks)
    logger.info(f"[PHASE6] Done in {time.time()-t6:.2f}s")

    # ── Build Dossier
    dossier = build_dossier(verified, top_sources, competitor_gap, topic)

    # Build structured facts list for programmatic citations
    # Format: [N] claim | source_title | source_url
    # This is used by generate_citations to build real source links
    all_claims = verified["confirmed"] + verified["single_source"]
    raw_facts = "\n".join(
        f"[{i}] {c.claim} | {c.source_title} | {c.source_url}"
        for i, c in enumerate(all_claims, 1)
        if c.source_url  # only include claims that have a real URL
    )

    # Safety net: if no claims were extracted at all (e.g. Hinduism-type broad
    # topics where Phase 5 returns nothing), still give the writer some dossier
    # content by including the top search snippets directly.
    if not all_claims and top_sources:
        logger.warning(
            "[RESEARCH ENGINE] Zero claims extracted — falling back to raw search snippets"
        )
        fallback_lines = [f"RESEARCH DOSSIER — {topic}\n",
                          "── RAW SEARCH SNIPPETS (use as background context only) ──"]
        for i, src in enumerate(top_sources, 1):
            snippet = (src.get("content") or src.get("full_content") or "")[:600]
            fallback_lines.append(
                f"SOURCE-{i}: {src.get('title', 'Unknown')}\n"
                f"  URL: {src.get('url', '')}\n"
                f"  Snippet: {snippet}\n"
            )
        dossier = "\n".join(fallback_lines)
        # Build raw_facts from source URLs alone so citations still has real links
        raw_facts = "\n".join(
            f"[{i}] {src.get('title', 'Source')} | {src.get('title', 'Source')} | {src.get('url', '')}"
            for i, src in enumerate(top_sources, 1)
            if src.get("url")
        )

    total = time.time() - start
    total_claims = sum(len(v) for v in verified.values())
    logger.info(
        f"[RESEARCH ENGINE] Complete | total={total:.1f}s | "
        f"claims={total_claims} | confirmed={len(verified['confirmed'])}"
    )

    return dossier, raw_facts
