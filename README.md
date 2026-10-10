# Blog Writer Agent

> **Live App → [blog-agent-rho.vercel.app](https://blog-agent-rho.vercel.app)**

AI blog posts all sound the same. Generic structure, cliché openers, no real research, conclusions that end with a question. This agent fixes that.

It's a 9-step pipeline that searches the live web, extracts real facts, writes a draft, critiques its own output, scans for 30+ known clichés, generates SEO metadata, and scores the result before you ever see it. The output reads like it was written by someone who actually researched the topic.

---

## Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        BLOG WRITER AGENT                        │
└─────────────────────────────────────────────────────────────────┘

  Topic + Audience + Length
          │
          ▼
  ┌───────────────┐
  │   1. PLAN     │  Generates outline, hook, sections, recommended length
  └───────┬───────┘
          │
          ▼
  ┌───────────────┐
  │  2. RESEARCH  │  4 parallel web searches via Tavily (advanced depth)
  └───────┬───────┘
          │
          ▼
  ┌───────────────┐
  │ 3. EXTRACT    │  Strips raw data → named sources, real stats, quotes only
  │    FACTS      │
  └───────┬───────┘
          │
          ▼
  ┌───────────────┐
  │  4. WRITE     │  Full blog using verified facts only. No hallucinations.
  │    DRAFT      │
  └───────┬───────┘
          │
          ▼
  ┌───────────────┐
  │ 5. CRITIQUE   │  Senior editor pass — fixes hooks, flow, weak conclusions
  │  + REWRITE    │
  └───────┬───────┘
          │
          ▼
  ┌───────────────┐
  │  6. CLICHÉ    │  Scans 30+ known clichés, replaces with topic-specific language
  │   DETECTOR    │
  └───────┬───────┘
          │
          ▼
  ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
  │  7. SEO META  │     │  8. EXTRAS    │     │  9. SCORE     │
  │               │     │  TL;DR        │     │  /10 with     │
  │  Title, desc, │     │  Pull Quote   │     │  5 dimensions │
  │  keywords     │     │  Key Takeaway │     │               │
  └───────────────┘     └───────────────┘     └───────────────┘
          │                     │                     │
          └─────────────────────┴─────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   FINAL BLOG OUTPUT   │
                    │   + PDF Download      │
                    └───────────────────────┘
```

---

## What Makes It Different

| Feature | This Agent | ChatGPT / Claude |
|---|---|---|
| Live web research | Yes — Tavily API | No — training data only |
| Fact extraction layer | Yes — verified facts only | No |
| Self-critique and rewrite | Yes — dedicated editor pass | No |
| Cliché detection | Yes — 30+ phrases banned | No |
| Quality scoring | Yes — 5 dimensions, /10 | No |
| TL;DR + Pull Quote | Yes — auto-generated | No |
| SEO metadata | Yes — title, desc, keywords | No |
| PDF export | Yes | No |

---

## Tech Stack

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Frontend  │    │   Backend   │    │     LLM     │    │  Research   │
│             │    │             │    │             │    │             │
│  HTML/CSS   │───▶│   FastAPI   │───▶│    Groq     │    │   Tavily    │
│  Vanilla JS │    │   Python    │    │  Cerebras   │    │  Advanced   │
│   Vercel    │    │   Render    │    │ LangChain   │    │   Search    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘

                                       ┌─────────────┐
                                       │   Memory    │
                                       │             │
                                       │   Chroma    │
                                       │  Vector DB  │
                                       └─────────────┘
```

---

## Prompt Engineering Highlights

**Temperature Strategy**
Different steps use different temperatures — not one setting for everything:
- `0.0` — Scorer (deterministic, consistent)
- `0.3` — Planner, Fact Extractor, SEO (precise, structured)
- `0.5` — Cliché Fixer (creative replacements)
- `0.6` — Writer, Editor (creative but controlled)

**Key Prompt Tricks**
- Writer persona: "contributor to The Atlantic, Wired, HBR, Vox" — sets a quality bar
- Banned phrases: "leverage", "synergy", "revolutionize" — forces real language
- The "so what" rule: after every stat, explain what it means for the reader
- Analogy requirement: at least one strong analogy per blog
- Banned openers: "In today's world..." and "Have you ever wondered..." — never
- No question endings: conclusions must end with a statement, not a question

---

## Setup

**1. Clone and install**
```bash
git clone https://github.com/Sujal-Kandi/Blog_Agent
cd Blog_Agent
pip install -r requirements.txt
```

**2. Create `.env`**
```
TAVILY_API_KEY=your_tavily_key
GROQ_API_KEY=your_groq_api_key
GROQ_API_KEY_2=your_groq_api_key_2
```

**3. Run CLI**
```bash
python main.py
```

**4. Run API**
```bash
uvicorn api:app --reload
```

---

## API

`POST /generate-blog`

```json
{
  "topic": "Why most diets fail in the first 30 days",
  "audience": "People aged 25-40 who keep falling off track",
  "length": "medium"
}
```

Length options: `short` (600-800 words) · `medium` (1000-1500 words) · `long` (2000-2500 words)

Response includes: `plan` · `draft` · `final_blog` · `extras` · `seo` · `scores`

---

## Example Output

**Blog Score**
```
Readability:        8  — Short paragraphs, clear language
Hook Strength:      9  — Opens with a specific stat
Content Depth:      8  — Named sources and real data throughout
SEO Strength:       7  — Focus keyword used naturally
Conclusion Quality: 8  — Ends with a specific, memorable line
Overall Score:      8.0 / 10
```

**TL;DR (auto-generated)**
- Ground roast coffee hit $8.41/lb in July 2025 — a 33% increase from a year ago
- Climate change in Brazil reduced crop yields, but commodity traders amplified the spike
- Specialty coffee demand is shifting costs upstream to farmers and consumers simultaneously

---

## Project Structure

```
Blog_Agent/
├── agent.py          # Full pipeline — all LLM functions
├── tools.py          # Tavily web search
├── memory.py         # Chroma vector store
├── main.py           # CLI with dry-run mode
├── api.py            # FastAPI endpoint
├── requirements.txt  # Dependencies
├── frontend/
│   ├── index.html    # UI
│   ├── style.css     # Dark minimal design
│   └── app.js        # Frontend logic
└── .env              # API keys (not committed)
```

---

## Deployment

- **Backend** — [Render](https://render.com) · Start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`
- **Frontend** — [Vercel](https://vercel.com) · Root directory: `frontend`
