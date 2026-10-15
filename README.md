# Blog Writer Agent

**Live app:** [blog-agent-rho.vercel.app](https://blog-agent-rho.vercel.app/)

Most generated blog posts sound generic. They use broad openings, repeat
obvious points, make unsupported claims, and end with a question.

Blog Writer Agent is a research-and-writing pipeline that turns a topic,
audience, and desired length into a structured blog draft. It searches the
live web, extracts research material, writes a draft, critiques and revises
it, detects common clichés, generates SEO metadata, and produces an
internal quality score.

It is a writing assistant, not an autonomous publisher. A human should
still review facts, sources, tone, and claims before publishing.

<img width="1800" height="950" alt="Screenshot 2026-10-14 232624" src="https://github.com/user-attachments/assets/5495e917-59cc-466c-9182-12d760251aef" />
<img width="1810" height="963" alt="Screenshot 2026-10-14 232507" src="https://github.com/user-attachments/assets/18c257e7-62a6-4efb-bc31-97f7cd0756d6" />


## Contents

- [What it produces](#what-it-produces)
- [The problem](#the-problem)
- [Pipeline](#pipeline)
- [What makes it different](#what-makes-it-different)
- [Prompt design](#prompt-design)
- [Example score output](#example-score-output)
- [API key rotation](#api-key-rotation)
- [Tech stack](#tech-stack)
- [Run locally](#run-locally)
- [API](#api)
- [Project structure](#project-structure)
- [Deployment](#deployment)
- [Project status](#project-status)
- [Limitations](#limitations)
- [Next work](#next-work)

## What it produces

For a given topic, the pipeline returns:

- A research-informed blog draft.
- An outline and recommended structure.
- A TL;DR.
- A pull quote.
- A key takeaway.
- SEO title, description, and keywords.
- A five-dimension internal quality score.
- A downloadable PDF.

The app supports three length settings:

| Length | Approximate word count |
|---|---:|
| Short | 600–800 words |
| Medium | 1,000–1,500 words |
| Long | 2,000–2,500 words |

## The problem

A single prompt can generate fluent text quickly, but it often produces:

- Generic openings and conclusions.
- Claims without clear sourcing.
- Repetitive AI-style phrasing.
- Weak transitions between sections.
- No editorial revision before the draft is returned.
- No structured metadata for publishing.

This project addresses those issues by separating planning, research,
fact extraction, writing, editing, and quality checks into distinct
stages.

It does not claim to eliminate hallucinations. It reduces some common
failure modes by grounding the writing stage in extracted research
material and adding review passes.

## Pipeline

```text
Topic + audience + length
          |
          v
1. Plan
   Outline, hook direction, sections, recommended length
          |
          v
2. Research
   Four parallel web searches through Tavily
          |
          v
3. Extract facts
   Identify named sources, statistics, and quotations
          |
          v
4. Write draft
   Generate the blog using the extracted research material
          |
          v
5. Critique and rewrite
   Improve hook, flow, transitions, and conclusion
          |
          v
6. Cliché detection
   Scan for common generic phrases and suggest replacements
          |
          v
7. SEO metadata
   Title, description, and keywords
          |
          v
8. Extras
   TL;DR, pull quote, and key takeaway
          |
          v
9. Score
   Five-dimension internal quality score
          |
          v
Final blog output and PDF download
```

### 1. Plan

The planner creates an outline, proposes a hook direction, defines
sections, and recommends a length based on the requested output size.

### 2. Research

The research stage runs four parallel Tavily searches using advanced
search depth.

Search results provide raw material for the next stage. They are not
automatically correct, complete, or sufficient for every topic.

### 3. Extract facts

The fact extractor identifies usable research material from the search
results, including:

- Named sources.
- Statistics.
- Quotations.
- Relevant factual claims.

This gives the writer a more structured evidence pool than raw search
snippets.

The extraction stage improves traceability, but it does not independently
verify every extracted statement against a primary source.

### 4. Write draft

The writer generates a full blog draft using the extracted research
material.

The prompt encourages specific evidence, clear explanations, and a
reader-focused structure. It also restricts several common generic
phrases and opening patterns.

### 5. Critique and rewrite

A separate editor pass reviews the draft for:

- Hook strength.
- Flow and transitions.
- Weak or vague conclusions.
- Readability.
- Overall structure.

It then produces a revised draft.

### 6. Cliché detection

The cliché detector scans for a curated list of common generic phrases
and suggests topic-specific replacements.

This pass improves style consistency. It does not make a claim more
factually accurate.

### 7–9. Metadata, extras, and scoring

The final stages generate:

- SEO title, description, and keywords.
- TL;DR.
- Pull quote.
- Key takeaway.
- A five-dimension quality score.

The score is an internal editorial heuristic. It is useful for comparing
drafts and identifying weak areas, but it is not an objective measure of
factual accuracy or publishing readiness.

## What makes it different

| Capability | Blog Writer Agent |
|---|---|
| Live web research | Uses Tavily search during generation |
| Research extraction | Separates facts, sources, and quotations before writing |
| Editorial revision | Includes a critique-and-rewrite pass |
| Style checks | Detects selected clichés and generic phrasing |
| Publishing extras | Generates TL;DR, pull quote, key takeaway, and SEO metadata |
| Quality feedback | Produces an internal multi-dimension score |
| Export | Provides PDF download |

This is not a claim that other models cannot perform research, editing,
or SEO tasks. The distinction is that this project packages those steps
into one explicit workflow with separate prompts, controls, and outputs.

## Prompt design

Different stages use different temperature settings:

| Stage | Temperature | Reason |
|---|---:|---|
| Scorer | 0.0 | More consistent scoring behaviour |
| Planner, fact extractor, SEO | 0.3 | Structured, precise output |
| Cliché fixer | 0.5 | Allows controlled rephrasing |
| Writer, editor | 0.6 | Creative but constrained writing |

Temperature zero reduces randomness; it does not guarantee identical
output across every run, provider version, or model update.

### Writing constraints

The prompts include rules such as:

- Explain the significance of a statistic rather than only stating it.
- Use at least one relevant analogy where appropriate.
- Avoid selected generic phrases and clichéd openings.
- End the conclusion with a statement rather than a question.
- Write for the specified audience rather than a generic reader.

These constraints shape style and structure. They do not guarantee
factual correctness.

## Example score output

```text
Readability:         8 — Short paragraphs, clear language
Hook Strength:       9 — Opens with a specific statistic
Content Depth:       8 — Named sources and data throughout
SEO Strength:        7 — Focus keyword used naturally
Conclusion Quality:  8 — Ends with a specific, memorable line

Overall Score:       8.0 / 10
```

This is an internal quality assessment generated by the pipeline. Treat
it as feedback for revision, not as proof of accuracy or search ranking.

## API key rotation

The pipeline supports multiple Groq API keys to reduce interruptions from
provider rate limits.

Keys are read from environment variables using the `GROQ_API_KEY` naming
pattern:

```env
GROQ_API_KEY=...
GROQ_API_KEY_2=...
GROQ_API_KEY_3=...
```

When one key is rate-limited, the pipeline can rotate to another
available configured key.

The number of keys is flexible: add more `GROQ_API_KEY_*` variables as
needed. This is a resilience measure, not unlimited capacity—provider
quotas, account limits, and usage policies still apply.

## Tech stack

| Layer | Tools |
|---|---|
| Frontend | HTML, CSS, vanilla JavaScript, Vercel |
| Backend | Python, FastAPI, Render |
| LLMs | Groq, Cerebras, LangChain |
| Research | Tavily |
| Memory | Chroma vector database |
| Export | PDF generation |

## Run locally

### 1. Clone and install

```bash
git clone [https://github.com/Sujal-Kandi/Blog_Agent.git](https://github.com/Sujal-Kandi/Blog_Agent.git)
cd Blog_Agent
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a local `.env` file:

```env
TAVILY_API_KEY=your_tavily_key

GROQ_API_KEY=your_first_groq_key
GROQ_API_KEY_2=your_second_groq_key
GROQ_API_KEY_3=your_third_groq_key
```

The project supports multiple Groq API keys for rate-limit rotation.

Add keys using the `GROQ_API_KEY` naming pattern:

```env
GROQ_API_KEY_4=your_fourth_groq_key
GROQ_API_KEY_5=your_fifth_groq_key
```

You can configure as many keys as your setup requires. The rotation logic
uses the available configured keys rather than requiring a fixed number.

Do not commit real API keys.

### 3. Run the CLI

```bash
python main.py
```

### 4. Run the API

```bash
uvicorn api:app --reload
```

## API

### `POST /generate-blog`

Request:

```json
{
  "topic": "Why most diets fail in the first 30 days",
  "audience": "People aged 25-40 who keep falling off track",
  "length": "medium"
}
```

The response includes:

```json
{
  "plan": {},
  "draft": {},
  "final_blog": {},
  "extras": {},
  "seo": {},
  "scores": {}
}
```

| Field | Contents |
|---|---|
| `plan` | Outline, hook direction, sections, and length recommendation |
| `draft` | Initial generated draft |
| `final_blog` | Revised blog after critique and rewriting |
| `extras` | TL;DR, pull quote, and key takeaway |
| `seo` | Title, description, and keywords |
| `scores` | Five-dimension internal quality score |

## Project structure

```text
Blog_Agent/
├── agent.py          # Pipeline and LLM functions
├── tools.py          # Tavily web search
├── memory.py         # Chroma vector store
├── main.py           # CLI, including dry-run mode
├── api.py            # FastAPI endpoint
├── requirements.txt  # Dependencies
└── frontend/
    ├── index.html    # Web interface
    ├── style.css     # Styling
    └── app.js        # Frontend logic
```

## Deployment

- **Backend:** Render  
  Start command:

  ```bash
  uvicorn api:app --host 0.0.0.0 --port $PORT
  ```

- **Frontend:** Vercel  
  Root directory: `frontend`

The web app supports PDF download for the generated blog.

## Project status

This project is actively developed. The core workflow—planning, research,
fact extraction, drafting, critique, cliché detection, metadata generation,
scoring, and PDF export—is implemented.

Some implementation details and edge cases are still being revisited. This
README documents the current intended behaviour and known limitations
rather than presenting the project as fully production-hardened.

## Limitations

- Search results may be incomplete, outdated, or low quality.
- Extracted facts are not independently verified against primary sources.
- The writer can still make unsupported or imprecise claims.
- The quality score is an internal heuristic, not an accuracy measure.
- SEO metadata does not guarantee search performance.
- API key rotation reduces rate-limit interruptions but does not remove provider quotas.
- Generated drafts require human review before publication.
- API key rotation reduces rate-limit interruptions but does not remove provider quotas.

## Next work

- Add source-link checking for extracted facts.
- Compare drafts against primary sources where available.
- Add human feedback into the revision loop.
- Evaluate output quality across more topics and audiences.
- Measure how often generated claims are supported by retrieved sources.

Built by Sujal Kandi.
