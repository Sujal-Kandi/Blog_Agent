"""
Agent Evaluation Suite for Blog Writer Agent
=============================================
Covers 3 types of evals:
  1. Deterministic checks  — fast, no LLM needed
  2. LLM-as-judge          — uses your LLM to score output quality
  3. LangSmith logging     — all results pushed to LangSmith for tracking

Run with:
    python eval.py
"""

import json
import time
from dotenv import load_dotenv

load_dotenv()

from agent import _invoke, CLICHES
from graph import run_blog_pipeline
from langsmith import traceable


# ── Test cases — topic + audience + what we expect ───────────────────────────
TEST_CASES = [
    {
        "topic": "Why most people quit the gym after January",
        "audience": "general readers",
        "length": "short",
        "expected_min_words": 500,
        "expected_max_words": 900,
    },
    {
        "topic": "The hidden cost of free shipping",
        "audience": "online shoppers",
        "length": "medium",
        "expected_min_words": 900,
        "expected_max_words": 1600,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# TYPE 1: DETERMINISTIC CHECKS
# These never call an LLM — pure code logic, always fast
# ─────────────────────────────────────────────────────────────────────────────

def check_word_count(blog: str, min_words: int, max_words: int) -> dict:
    """Check if blog is within expected word count range."""
    count = len(blog.split())
    passed = min_words <= count <= max_words
    return {
        "check": "word_count",
        "passed": passed,
        "value": count,
        "expected": f"{min_words}–{max_words}",
        "message": f"{count} words" + ("" if passed else f" (expected {min_words}–{max_words})")
    }


def check_no_cliches(blog: str) -> dict:
    """Check that no banned clichés made it into the final blog."""
    found = [c for c in CLICHES if c.lower() in blog.lower()]
    passed = len(found) == 0
    return {
        "check": "no_cliches",
        "passed": passed,
        "value": found,
        "message": "No clichés found" if passed else f"Found clichés: {found}"
    }


def check_has_title(blog: str) -> dict:
    """Check that the blog has a title (first non-empty line, reasonable length)."""
    lines = [l.strip() for l in blog.strip().split("\n") if l.strip()]
    first_line = lines[0] if lines else ""
    passed = 10 <= len(first_line) <= 120
    return {
        "check": "has_title",
        "passed": passed,
        "value": first_line[:80],
        "message": "Title looks good" if passed else f"Suspicious title: '{first_line[:80]}'"
    }


def check_no_hashtags(blog: str) -> dict:
    """Check that the blog contains no hashtags or markdown headers."""
    has_hashtag = any(line.strip().startswith("#") for line in blog.split("\n"))
    return {
        "check": "no_hashtags",
        "passed": not has_hashtag,
        "message": "No hashtags found" if not has_hashtag else "Found hashtag/markdown headers"
    }


def check_seo_fields(seo: dict) -> dict:
    """Check that all required SEO fields are present and non-empty."""
    required = ["Meta Title", "Meta Description", "Focus Keyword", "Suggested Tags"]
    missing = [f for f in required if not seo.get(f, "").strip()]
    passed = len(missing) == 0
    return {
        "check": "seo_fields",
        "passed": passed,
        "message": "All SEO fields present" if passed else f"Missing SEO fields: {missing}"
    }


def check_scores_present(scores: dict) -> dict:
    """Check that the agent returned all expected score fields."""
    required = ["Readability", "Hook Strength", "Content Depth", "Overall Score"]
    missing = [f for f in required if f not in scores]
    passed = len(missing) == 0
    return {
        "check": "scores_present",
        "passed": passed,
        "message": "All score fields present" if passed else f"Missing score fields: {missing}"
    }


def run_deterministic_checks(result: dict, min_words: int, max_words: int) -> list:
    """Run all deterministic checks on a pipeline result."""
    blog = result.get("final_blog", "")
    return [
        check_word_count(blog, min_words, max_words),
        check_no_cliches(blog),
        check_has_title(blog),
        check_no_hashtags(blog),
        check_seo_fields(result.get("seo", {})),
        check_scores_present(result.get("scores", {})),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# TYPE 2: LLM-AS-JUDGE
# Use your LLM to evaluate output quality — scores 1-10
# ─────────────────────────────────────────────────────────────────────────────

@traceable(name="llm_judge_eval")
def llm_judge(blog: str, topic: str, audience: str) -> dict:
    """
    Ask the LLM to evaluate the blog on 4 dimensions.
    Returns a dict of scores and reasoning.
    """
    prompt = f"""
You are an expert content evaluator. Score this blog objectively from 1 to 10 on each dimension.
Be strict — a 10 means it could be published in The Atlantic today.

Topic: {topic}
Audience: {audience}

Blog:
{blog[:3000]}

Score each dimension with a number and one-line reason:

OnTopic: (does the blog actually address the topic? 1=completely off, 10=perfectly on topic)
HookQuality: (does the first paragraph grab attention immediately? 1=boring, 10=compelling)
Specificity: (does it use real data, named examples, specific facts? 1=vague, 10=very specific)
AudienceFit: (is the tone and content right for the stated audience? 1=wrong audience, 10=perfect fit)

Overall: (average of the 4 scores above)
Verdict: (one sentence — what works and what would make it better)

Return each on its own line with label followed by colon. Nothing else.
"""
    raw = _invoke(prompt, temperature=0.0)
    scores = {}
    for line in raw.strip().split("\n"):
        if ":" in line:
            key, _, value = line.partition(":")
            scores[key.strip()] = value.strip()
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# TYPE 3: FULL EVAL RUN + LANGSMITH LOGGING
# ─────────────────────────────────────────────────────────────────────────────

@traceable(name="run_eval_case")
def run_eval_case(test_case: dict) -> dict:
    """Run a single test case through the full pipeline and evaluate it."""
    case_start = time.time()
    topic = test_case["topic"]
    audience = test_case["audience"]
    length = test_case["length"]

    print(f"\n{'='*60}")
    print(f"EVAL: {topic[:55]}")
    print(f"{'='*60}")

    # Run the pipeline
    start = time.time()
    result = run_blog_pipeline(topic=topic, audience=audience, length=length)
    duration = round(time.time() - start, 1)
    print(f"Pipeline completed in {duration}s")

    # Type 1: Deterministic checks
    det_checks = run_deterministic_checks(
        result,
        test_case["expected_min_words"],
        test_case["expected_max_words"]
    )
    det_passed = sum(1 for c in det_checks if c["passed"])
    det_total = len(det_checks)

    print(f"\nDeterministic checks: {det_passed}/{det_total} passed")
    for c in det_checks:
        status = "✅" if c["passed"] else "❌"
        print(f"  {status} {c['check']}: {c['message']}")

    # Type 2: LLM-as-judge
    print("\nRunning LLM judge...")
    judge_start = time.time()
    llm_scores = llm_judge(result["final_blog"], topic, audience)
    judge_duration = round(time.time() - judge_start, 1)
    print(f"  Overall: {llm_scores.get('Overall', 'N/A')}")
    print(f"  Verdict: {llm_scores.get('Verdict', 'N/A')}")
    print(f"  Judge completed in {judge_duration}s")

    # Agent's own self-score (from the pipeline)
    agent_score = result.get("scores", {}).get("Overall Score", "N/A")
    print(f"  Agent self-score: {agent_score}")

    eval_result = {
        "topic": topic,
        "audience": audience,
        "length": length,
        "duration_seconds": duration,
        "judge_duration_seconds": judge_duration,
        "total_duration_seconds": round(time.time() - case_start, 1),
        "word_count": len(result["final_blog"].split()),
        "deterministic": {
            "passed": det_passed,
            "total": det_total,
            "checks": det_checks,
        },
        "llm_judge": llm_scores,
        "agent_self_score": agent_score,
    }

    return eval_result


def run_all_evals():
    """Run all test cases and print a summary."""
    print("\n🧪 Starting Blog Agent Eval Suite")
    print(f"Running {len(TEST_CASES)} test cases...\n")

    all_results = []
    for test_case in TEST_CASES:
        result = run_eval_case(test_case)
        all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("EVAL SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        det = r["deterministic"]
        llm_overall = r["llm_judge"].get("Overall", "N/A")
        print(f"\n📝 {r['topic'][:50]}")
        print(f"   Deterministic: {det['passed']}/{det['total']} passed")
        print(f"   LLM Judge score: {llm_overall}")
        print(f"   Agent self-score: {r['agent_self_score']}")
        print(
            f"   Pipeline: {r['duration_seconds']}s | "
            f"Judge: {r['judge_duration_seconds']}s | "
            f"Total: {r['total_duration_seconds']}s | Words: {r['word_count']}"
        )

    # Save results to file
    with open("eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved to eval_results.json")
    print("✅ All traces logged to LangSmith under project: Blog_Agent")


if __name__ == "__main__":
    run_all_evals()
