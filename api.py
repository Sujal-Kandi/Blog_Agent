import os
import time
import logging
import sentry_sdk
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from graph import run_blog_pipeline

# ── Sentry setup ──────────────────────────────────────────────────────────────
_sentry_dsn = os.getenv("SENTRY_DSN", "")
sentry_sdk.init(
    dsn=_sentry_dsn,
    traces_sample_rate=0.1,
    environment="production",
    send_default_pii=False
)
if _sentry_dsn:
    print("Sentry initialized successfully")
else:
    print("SENTRY_DSN not set - Sentry is disabled")

# ── Logging setup ─────────────────────────────────────────────────────────────
log_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("blog_api")
logger.setLevel(logging.INFO)

# Console handler — shows in terminal and Render logs
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# File handler — saves to blog_api.log locally
file_handler = logging.FileHandler("blog_api.log")
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

app = FastAPI(title="Blog Writer Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class BlogRequest(BaseModel):
    topic: str
    audience: Optional[str] = "general readers"
    length: Optional[str] = "medium"


# ── Input validation ──────────────────────────────────────────────────────────

def _is_gibberish(text: str) -> bool:
    """
    Returns True if the text looks like random characters.
    Checks two signals:
      1. Vowel ratio — real words in English have at least 25% vowels
      2. Unique character ratio — gibberish like 'asdfghjkl' has too many unique chars
         relative to its length, while real words reuse letters
    """
    cleaned = text.lower().replace(" ", "")
    if not cleaned:
        return True

    vowels = sum(1 for c in cleaned if c in "aeiou")
    vowel_ratio = vowels / len(cleaned)

    unique_ratio = len(set(cleaned)) / len(cleaned)

    # Too few vowels (< 15%) AND too many unique chars (> 70%) = gibberish
    return vowel_ratio < 0.15 and unique_ratio > 0.70


def validate_inputs(topic: str, audience: str) -> None:
    """
    Validates topic and audience before the pipeline runs.
    Raises HTTPException with a clear message on any failure.
    """
    topic = topic.strip()
    audience = audience.strip()

    # ── Topic checks ──────────────────────────────────────────────────────────
    if not topic:
        raise HTTPException(status_code=422, detail="Topic cannot be empty.")

    words = topic.split()
    if len(words) < 3:
        raise HTTPException(
            status_code=422,
            detail="Topic is too short. Please use at least 3 words — e.g. 'Why coffee prices are rising'."
        )

    if topic.replace(" ", "").isdigit():
        raise HTTPException(
            status_code=422,
            detail="Topic cannot be just numbers. Please describe what you want to write about."
        )

    if _is_gibberish(topic):
        raise HTTPException(
            status_code=422,
            detail="Topic looks like random characters. Please enter a real topic — e.g. 'Why people quit their jobs'."
        )

    if len(topic) > 300:
        raise HTTPException(
            status_code=422,
            detail="Topic is too long. Keep it under 300 characters."
        )

    # ── Audience checks ───────────────────────────────────────────────────────
    if not audience:
        raise HTTPException(status_code=422, detail="Audience cannot be empty.")

    if _is_gibberish(audience):
        raise HTTPException(
            status_code=422,
            detail="Audience looks like random characters. Use something like 'Indians', 'Adults', or 'General readers'."
        )

    if len(audience) > 100:
        raise HTTPException(
            status_code=422,
            detail="Audience description is too long. Keep it under 100 characters."
        )


@app.post("/generate-blog")
def generate_blog(request: BlogRequest):
    topic   = (request.topic or "").strip()
    audience = (request.audience or "general readers").strip()
    length   = (request.length or "medium").strip().lower()

    # Validate before touching the pipeline
    validate_inputs(topic, audience)

    if length not in ("short", "medium", "long"):
        raise HTTPException(
            status_code=422,
            detail="Length must be 'short', 'medium', or 'long'."
        )

    start = time.time()
    logger.info(f"[REQUEST] topic='{topic[:60]}' | audience='{audience[:40]}' | length={length}")
    try:
        result = run_blog_pipeline(
            topic=topic,
            audience=audience,
            length=length,
        )
        duration = time.time() - start
        word_count = len(result.get("final_blog", "").split())
        score = result.get("scores", {}).get("Overall Score", "N/A")
        logger.info(f"[SUCCESS] topic='{topic[:60]}' | duration={duration:.1f}s | words={word_count} | score={score}")
        return result
    except HTTPException:
        raise  # re-raise validation errors as-is
    except Exception as e:
        duration = time.time() - start
        msg = str(e).lower()
        logger.error(f"[FAILED] topic='{topic[:60]}' | duration={duration:.1f}s | error={str(e)[:120]}")
        if "rate limit" in msg or "rate limited" in msg or "429" in msg:
            raise HTTPException(
                status_code=429,
                detail="API rate limit reached. Please try again in 5 minutes."
            )
        sentry_sdk.capture_exception(e)
        raise HTTPException(
            status_code=500,
            detail="Something went wrong. Please try again."
        )


