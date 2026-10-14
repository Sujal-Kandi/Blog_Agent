# Blog Writer Agent

Most generated blog posts are generic: a broad introduction, a list of
obvious points, a few unsupported claims, and a conclusion that ends with
a question.

Blog Writer Agent is a research-and-writing pipeline that turns a topic,
audience, and desired length into a structured blog draft. It searches the
live web, extracts source-linked facts, writes a draft, critiques and
revises it, detects common clichés, generates SEO metadata, and produces
a quality score.

It is a writing assistant, not an autonomous publisher. A human should
still review facts, sources, tone, and claims before publishing.

**Live app:** [blog-agent-rho.vercel.app](https://blog-agent-rho.vercel.app/)

<img width="1810" height="963" alt="{ABBA857E-8A5A-421E-B56E-8ABD32A168F6}" src="https://github.com/user-attachments/assets/28ed173d-9787-402f-b43f-48be49c98ed1" />
<img width="1800" height="950" alt="{B8DC707E-B22D-4F16-A807-E1F6A067088C}" src="https://github.com/user-attachments/assets/5d062bd5-452b-4c37-bf05-4be5b49b2dc4" />





## What it produces

For a given topic, the pipeline returns:

- A research-backed blog draft.
- An outline and recommended structure.
- A TL;DR.
- A pull quote.
- A key takeaway.
- SEO title, description, and keywords.
- A five-dimension internal quality score.
- A downloadable PDF.

The app accepts short, medium, and long length settings:

| Length | Approximate word count |
|---|---|
| Short | 600–800 words |
| Medium | 1,000–1,500 words |
| Long | 2,000–2,500 words |

## The problem

A single prompt to an LLM can produce fluent text quickly, but it often
has predictable weaknesses:

- Generic openings and conclusions.
- Claims without clear sourcing.
- Repetitive AI-style phrasing.
- Weak transitions between sections.
- No editorial pass before the draft is returned.
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
   Improve hook, flow, section transitions, and conclusion
          |
          v
6. Cliché detection
   Scan for common generic phrases and suggest replacements
          |
          v
7-9. Metadata, extras, and scoring
   SEO fields, TL;DR, pull quote, key takeaway, quality score
          |
          v
Final blog output and PDF export
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
GROQ_API_KEY=your_groq_api_key
GROQ_API_KEY_2=your_second_groq_key
```

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

Response fields include:

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

## Limitations

- Search results may be incomplete, outdated, or low quality.
- Extracted facts are not independently verified against primary sources.
- The writer can still make unsupported or imprecise claims.
- The quality score is heuristic, not an accuracy measure.
- SEO metadata does not guarantee search performance.
- Generated drafts require human review before publication.

## Next work

- Add source-link checking for extracted facts.
- Compare drafts against primary sources where available.
- Add human feedback into the revision loop.
- Evaluate output quality across more topics and audiences.
- Measure how often generated claims are supported by retrieved sources.

Built by Sujal Kandi.
