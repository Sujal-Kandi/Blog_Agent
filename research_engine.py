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


class ClaimRegistryEntry(BaseModel):
    """
    A single registered claim the writer is allowed to cite.
    Every specific stat, quote, or named event MUST have one of these.
    The writer references claims by ID — not by memory.
    """
    id: str              # e.g. "CLAIM-001"
    claim: str           # the full claim text
    claim_type: str      # "stat" | "quote" | "event" | "general"
    confidence: str      # "confirmed" | "single_source" | "disputed"
    source_title: str
    source_url: str
    # Pre-extracted forbidden patterns that exist in this claim —
    # used by the filter to know which pattern instances ARE grounded.
    forbidden_matches: list[str]  # e.g. ["103.07", "2,600 runs", "2013 Champions Trophy"]


class ClaimRegistry:
    """
    Structured registry of every verified claim from the research engine.
    Passed alongside the dossier string into write_blog and the filter.

    Key operations:
      - build()         called by build_dossier() — assigns IDs to all claims
      - to_writer_block()  compact string injected into write_blog prompt
      - all_grounded_values()  flat set of every number/event already registered
                               — used by forbidden_pattern_filter for O(1) lookup
    """

    def __init__(self):
        self.entries: list[ClaimRegistryEntry] = []
        self._grounded_values: set[str] | None = None   # lazy cache

    def build(self, verified: dict[str, list["Claim"]]) -> None:
        """Populate registry from verify_claims() output."""
        self.entries.clear()
        self._grounded_values = None
        counter = 1
        for confidence_level in ("confirmed", "single_source", "disputed"):
            for claim in verified.get(confidence_level, []):
                entry = self._make_entry(claim, confidence_level, counter)
                self.entries.append(entry)
                counter += 1

    def _make_entry(self, claim: "Claim", confidence: str, n: int) -> ClaimRegistryEntry:
        import re as _re
        cid = f"CLAIM-{n:03d}"

        # Extract the specific values in this claim that would match
        # the forbidden patterns — so the filter knows they're grounded.
        text = claim.claim
        forbidden_matches: list[str] = []

        # Numbers with units
        for m in _re.finditer(
            r'\b\d[\d,]*(?:\.\d+)?\s*(?:%|runs|wickets|matches|years|km|mph|goals|titles|sixes|balls|overs)?\b',
            text
        ):
            val = m.group(0).strip()
            if val:
                forbidden_matches.append(val.lower())

        # Named events: "2013 Champions Trophy", "2011 World Cup", etc.
        for m in _re.finditer(
            r'\b\d{4}\s+(?:ICC\s+)?(?:IPL|World Cup|Champions Trophy|T20I|T20\s+World Cup|'
            r'ODI|Test|Grand Prix|Premier League|NBA|Super Bowl|Ashes|series|final|semi-final|'
            r'quarter-final)\b',
            text, _re.IGNORECASE
        ):
            forbidden_matches.append(m.group(0).lower())

        # Specific scores: "91 off 61 balls", "scored 90"
        for m in _re.finditer(r'\b(?:scored?|made|hit)\s+\d+', text, _re.IGNORECASE):
            forbidden_matches.append(m.group(0).lower())

        return ClaimRegistryEntry(
            id=cid,
            claim=text,
            claim_type=claim.claim_type,
            confidence=confidence,
            source_title=claim.source_title,
            source_url=claim.source_url,
            forbidden_matches=list(set(forbidden_matches)),
        )

    def to_writer_block(self) -> str:
        """
        Compact string for the write_blog prompt.
        Shows each claim with its ID and confidence level.
        Writer must cite by CLAIM-ID for any specific stat/event/quote.
        """
        if not self.entries:
            return "CLAIM REGISTRY: empty — no verified claims available. Write without specific numbers."

        lines = ["═══ CLAIM REGISTRY — CITE BY ID ═══",
                 "Every specific number, score, event, or quote below has a CLAIM-ID.",
                 "To use a claim: write it and add (CLAIM-ID) at the end of the sentence.",
                 "If a claim has no CLAIM-ID here — describe the PATTERN, not the specific instance.",
                 ""]

        confirmed   = [e for e in self.entries if e.confidence == "confirmed"]
        single      = [e for e in self.entries if e.confidence == "single_source"]
        disputed    = [e for e in self.entries if e.confidence == "disputed"]

        if confirmed:
            lines.append("── CONFIRMED (safe to state as fact) ──")
            for e in confirmed:
                lines.append(f"  [{e.id}] [{e.claim_type.upper()}] {e.claim}")
                lines.append(f"         Source: {e.source_title}")
            lines.append("")

        if single:
            lines.append('── SINGLE-SOURCE (must write: "according to [source]...") ──')
            for e in single:
                lines.append(f"  [{e.id}] [{e.claim_type.upper()}] {e.claim}")
                lines.append(f"         Source: {e.source_title}")
            lines.append("")

        if disputed:
            lines.append('── DISPUTED (must write: "some sources suggest... while others...") ──')
            for e in disputed:
                lines.append(f"  [{e.id}] [{e.claim_type.upper()}] {e.claim}")
            lines.append("")

        lines.append("══════════════════════════════════")
        return "\n".join(lines)

    def all_grounded_values(self) -> set[str]:
        """
        Returns flat set of every number/event/score already in the registry.
        Used by forbidden_pattern_filter for O(1) lookup — built once, cached.
        """
        if self._grounded_values is not None:
            return self._grounded_values
        result: set[str] = set()
        for entry in self.entries:
            for val in entry.forbidden_matches:
                result.add(val.lower().strip())
            # Also add the raw numbers from the claim text directly
            import re as _re
            for m in _re.finditer(r'\b\d[\d,]*(?:\.\d+)?\b', entry.claim):
                result.add(m.group(0).replace(",", "").lower())
        self._grounded_values = result
        return result


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
    Check if a claim's key number/name appears in 2+ chunks from DIFFERENT URLs,
    AND that the number appears in the SAME CONTEXT as the claim (not just anywhere).

    Stricter than before:
    - Numbers must appear within 60 chars of a claim keyword in the chunk text
    - Both chunks must be from different domains (not just different URLs)
    - Short numbers (1-2 digits) require a context window match, not bare match
    """
    numbers = re.findall(r'\b\d+\.?\d*\b', claim.claim)
    # Key non-stopword words from the claim for context anchoring
    stopwords = {"the", "a", "an", "is", "are", "of", "in", "his", "her",
                 "with", "for", "and", "that", "this", "was", "has", "had"}
    claim_words = [
        w.lower() for w in re.findall(r'\b[a-zA-Z]{4,}\b', claim.claim)
        if w.lower() not in stopwords
    ][:4]

    if not numbers:
        # No numbers — fall back to keyword overlap
        supporting_urls = set()
        for chunk in chunks:
            chunk_lo = chunk["text"].lower()
            if sum(1 for w in claim_words if w in chunk_lo) >= 2:
                supporting_urls.add(chunk["source_url"])
        return len(supporting_urls) >= 2

    supporting_domains: set[str] = set()

    for chunk in chunks:
        chunk_lo = chunk["text"].lower()
        url = chunk.get("source_url", "")

        # Extract domain for deduplication
        try:
            domain = url.split("/")[2].lower().replace("www.", "")
        except IndexError:
            domain = url

        matched = False
        for num in numbers:
            num_lo = num.lower()

            # For short numbers (≤2 digits), require them to appear near
            # a claim keyword within a 80-char window — prevents "88" matching
            # unrelated "188" or "880" in the text
            if len(num_lo.replace(".", "")) <= 2:
                # Find positions of this number in the chunk
                for m in re.finditer(r'\b' + re.escape(num_lo) + r'\b', chunk_lo):
                    start = max(0, m.start() - 80)
                    end   = min(len(chunk_lo), m.end() + 80)
                    window = chunk_lo[start:end]
                    # At least one claim keyword must appear in the window
                    if any(w in window for w in claim_words):
                        matched = True
                        break
            else:
                # Longer numbers: exact match is specific enough
                if re.search(r'\b' + re.escape(num_lo) + r'\b', chunk_lo):
                    # Still require at least one claim keyword nearby
                    for m in re.finditer(r'\b' + re.escape(num_lo) + r'\b', chunk_lo):
                        start = max(0, m.start() - 120)
                        end   = min(len(chunk_lo), m.end() + 120)
                        window = chunk_lo[start:end]
                        if any(w in window for w in claim_words):
                            matched = True
                            break

            if matched:
                break

        if matched:
            supporting_domains.add(domain)

    return len(supporting_domains) >= 2


def _extract_primary_number(claim_text: str) -> str | None:
    """
    Extract the most significant number from a claim — the one that defines it.
    e.g. "bowling average of 102.5" → "102.5"
         "3 wickets in 13 matches"  → "3"  (first specific stat)
    Returns None if no numbers found.
    """
    # Prefer decimal numbers (more specific)
    decimals = re.findall(r'\b\d+\.\d+\b', claim_text)
    if decimals:
        return decimals[0]
    # Fall back to any integer
    integers = re.findall(r'\b\d+\b', claim_text)
    return integers[0] if integers else None


def _check_contradiction(claim: Claim, chunks: list[dict]) -> bool:
    """
    Detect if other chunks contain a DIFFERENT value for the same stat.
    e.g. claim says "average of 88", another chunk says "average of 102.5"
    for the same player/metric = contradiction → mark as disputed.

    Returns True if a contradiction is found.
    """
    primary_num = _extract_primary_number(claim.claim)
    if not primary_num:
        return False

    stopwords = {"the", "a", "an", "is", "are", "of", "in", "his", "her",
                 "with", "for", "and", "that", "this", "was", "has", "had"}
    claim_words = [
        w.lower() for w in re.findall(r'\b[a-zA-Z]{4,}\b', claim.claim)
        if w.lower() not in stopwords
    ][:3]

    if not claim_words:
        return False

    # Metric keywords that signal the same stat is being described
    metric_words = re.findall(
        r'\b(average|economy|wickets?|strike\s+rate|runs?\s+per\s+(over|wicket)|appearances?)\b',
        claim.claim, re.IGNORECASE
    )
    metric_terms = [m[0].lower() for m in metric_words]

    for chunk in chunks:
        # Skip the claim's own source
        if chunk.get("source_url") == claim.source_url:
            continue

        chunk_lo = chunk["text"].lower()

        # Check if this chunk is about the same subject
        subject_match = sum(1 for w in claim_words if w in chunk_lo)
        if subject_match < 1:
            continue

        # Check if the same metric appears with a DIFFERENT number
        for metric in metric_terms:
            if metric not in chunk_lo:
                continue
            # Find numbers near this metric in the chunk
            for m in re.finditer(re.escape(metric), chunk_lo):
                start = max(0, m.start() - 40)
                end   = min(len(chunk_lo), m.end() + 40)
                window = chunk_lo[start:end]
                chunk_nums = re.findall(r'\b\d+\.?\d*\b', window)
                for cn in chunk_nums:
                    # Different number for same metric = contradiction
                    # Allow ±5% tolerance for rounding differences
                    try:
                        cv = float(cn)
                        pv = float(primary_num)
                        if pv > 0 and abs(cv - pv) / pv > 0.05 and cv != pv:
                            logger.info(
                                f"[PHASE6] Contradiction detected: "
                                f"claim={primary_num} vs chunk={cn} for metric='{metric}'"
                            )
                            return True
                    except ValueError:
                        continue

    return False


def _verify_with_search(claim: Claim) -> str:
    """
    Fire one targeted Tavily search to verify a single-source stat.
    Returns: 'confirmed' | 'disputed' | 'single_source'

    Stricter than before:
    - Numbers must appear in the SAME CONTEXT as the claim keywords
    - A different number for the same stat = 'disputed', not 'confirmed'
    - Short numbers require a context window, not a bare text match
    """
    primary_num = _extract_primary_number(claim.claim)
    claim_lo    = claim.claim.lower()

    stopwords = {"the", "a", "an", "is", "are", "of", "in", "his", "her",
                 "with", "for", "and", "that", "this", "was", "has", "had"}
    claim_words = [
        w for w in re.findall(r'\b[a-zA-Z]{4,}\b', claim_lo)
        if w not in stopwords
    ][:4]

    query = claim.claim[:120] if len(claim.claim) > 60 else f'"{claim.claim[:80]}"'

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=_get_tavily_key())
        response = client.search(query=query, search_depth="basic", max_results=4)
        results  = response.get("results", [])

        if not results:
            return "single_source"

        supporting    = 0
        contradicting = 0

        for r in results:
            # Skip the original source
            if claim.source_url and claim.source_url in r.get("url", ""):
                continue

            text = (r.get("content", "") + " " + r.get("title", "")).lower()

            if primary_num:
                # Find the primary number in this result text
                num_lo = primary_num.lower()

                # Require contextual match — number must appear near a claim keyword
                found_in_context = False
                contradicted     = False

                for m in re.finditer(r'\b' + re.escape(num_lo) + r'\b', text):
                    start  = max(0, m.start() - 100)
                    end    = min(len(text), m.end() + 100)
                    window = text[start:end]
                    if any(w in window for w in claim_words):
                        found_in_context = True
                        break

                # Also look for the same metric with a DIFFERENT number
                metric_words_in_claim = re.findall(
                    r'\b(average|economy|wickets?|strike\s+rate|runs?\s+per\s+(over|wicket)|appearances?)\b',
                    claim_lo, re.IGNORECASE
                )
                for metric_match in metric_words_in_claim:
                    metric = metric_match[0].lower()
                    if metric not in text:
                        continue
                    for mm in re.finditer(re.escape(metric), text):
                        ws  = max(0, mm.start() - 40)
                        we  = min(len(text), mm.end() + 40)
                        win = text[ws:we]
                        alt_nums = re.findall(r'\b\d+\.?\d*\b', win)
                        for an in alt_nums:
                            try:
                                av = float(an)
                                pv = float(primary_num)
                                if pv > 0 and abs(av - pv) / pv > 0.05 and av != pv:
                                    contradicted = True
                                    logger.info(
                                        f"[PHASE6-SEARCH] Contradiction in search result: "
                                        f"claim={primary_num} result_says={an} metric={metric}"
                                    )
                                    break
                            except ValueError:
                                continue
                        if contradicted:
                            break

                if contradicted:
                    contradicting += 1
                elif found_in_context:
                    supporting += 1

            else:
                # No number — keyword overlap check
                keyword_hits = sum(1 for w in claim_words if w in text)
                if keyword_hits >= 3:
                    supporting += 1

        # Decision logic:
        # contradictions > supporting → disputed
        # supporting >= 1 and no contradictions → confirmed
        # else → single_source
        if contradicting > 0 and contradicting >= supporting:
            return "disputed"
        if supporting >= 1:
            return "confirmed"
        return "single_source"

    except Exception as e:
        logger.warning(f"[PHASE6] Verification search failed: {e}")
        return "single_source"


def verify_claims(claims: list[Claim], chunks: list[dict]) -> dict[str, list[Claim]]:
    """
    Phase 6: Verify claims and bucket them into confirmed / single_source / disputed.

    Improvements over previous version:
    - _cross_check_in_chunks: contextual window match, domain deduplication
    - _check_contradiction: detects conflicting values for the same stat
    - _verify_with_search: contextual number matching + contradiction detection
    - Disputed claims from either check go to 'disputed' bucket
    """
    confirmed     = []
    single_source = []
    disputed      = []

    verification_searches    = 0
    MAX_VERIFICATION_SEARCHES = 4

    for claim in claims:
        if claim.claim_type in ("stat", "quote"):

            # Step 0: contradiction check within existing chunks (free, instant)
            if _check_contradiction(claim, chunks):
                disputed.append(claim)
                logger.info(f"[PHASE6] Disputed (contradiction): {claim.claim[:60]}")
                continue

            # Step 1: free cross-check — does this value appear in 2+ domains?
            if _cross_check_in_chunks(claim, chunks):
                confirmed.append(claim)
                continue

            # Step 2: paid search verification (capped)
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
) -> tuple[str, ClaimRegistry]:
    """
    Build the final research dossier string AND a ClaimRegistry for the writing engine.

    Returns:
        (dossier_str, claim_registry)
        dossier_str    — structured evidence string for write_blog prompt
        claim_registry — ClaimRegistry object for forbidden_pattern_filter and grounding_score
    """
    # ── Build ClaimRegistry first ─────────────────────────────────────────────
    registry = ClaimRegistry()
    registry.build(verified)

    lines = [f"RESEARCH DOSSIER — {topic}\n"]

    # Confirmed facts
    if verified["confirmed"]:
        lines.append("── CONFIRMED FACTS (verified by 2+ independent sources — safe to state directly) ──")
        for i, c in enumerate(verified["confirmed"], 1):
            source_context = f"{c.source_title}" if c.source_title else "unknown source"
            date_context = f" ({c.published_date[:4]})" if c.published_date and len(c.published_date) >= 4 else ""
            cid = f"CLAIM-{i:03d}"
            lines.append(
                f"{cid}: {c.claim}\n"
                f"  Context: {c.evidence[:300]}\n"
                f"  Source: {source_context}{date_context}\n"
                f"  URL: {c.source_url}\n"
                f"  Type: {c.claim_type}"
            )
        lines.append("")

    # Single-source facts — offset index by confirmed count
    confirmed_count = len(verified["confirmed"])
    if verified["single_source"]:
        lines.append(
            '── SINGLE-SOURCE FACTS (from ONE source only — MUST be attributed: '
            '"according to [Source]..." or "reportedly..." — NEVER state as confirmed fact) ──'
        )
        for i, c in enumerate(verified["single_source"], confirmed_count + 1):
            source_context = f"{c.source_title}" if c.source_title else "unknown source"
            date_context = f" ({c.published_date[:4]})" if c.published_date and len(c.published_date) >= 4 else ""
            cid = f"CLAIM-{i:03d}"
            lines.append(
                f"{cid} [SINGLE-SOURCE — attribute to {source_context!r}]: {c.claim}\n"
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

    # Hard rules for the writer
    lines.append(
        "── WRITER RULES ──\n"
        "1. Only use numbers and statistics from CONFIRMED FACTS above.\n"
        "2. SINGLE-SOURCE facts must be attributed to the source — never stated as plain fact.\n"
        "3. If the dossier has no confirmed stat for a claim, write the insight WITHOUT a number.\n"
        "   WRONG: 'his win-rate jumps to 80% when he scores a fifty'\n"
        "   RIGHT: 'India's win-rate rises sharply when he converts a fifty'\n"
        "4. Never invent percentages, run totals, averages, or match results not in this dossier.\n"
        "5. DISPUTED claims must be presented as disputed — do not pick one side.\n"
    )

    # Competitor gap
    if competitor_gap:
        lines.append("── COMPETITOR GAP (what existing articles miss — address this angle) ──")
        lines.append(competitor_gap)
        lines.append("")

    return "\n".join(lines), registry


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def run_research_engine(topic: str, competitor_gap: str, invoke_fn) -> tuple[str, str, "ClaimRegistry"]:
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
        return "", "", ClaimRegistry()

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

    # ── Build Dossier + ClaimRegistry
    dossier, registry = build_dossier(verified, top_sources, competitor_gap, topic)

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
        # Registry stays empty — forbidden filter will treat everything as ungrounded

    total = time.time() - start
    total_claims = sum(len(v) for v in verified.values())
    logger.info(
        f"[RESEARCH ENGINE] Complete | total={total:.1f}s | "
        f"claims={total_claims} | confirmed={len(verified['confirmed'])}"
    )

    return dossier, raw_facts, registry
