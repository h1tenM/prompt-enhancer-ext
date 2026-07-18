# ✨ Prompt Enhancer (HackBU 2026)

A Chrome extension that rewrites vague requests into well-structured prompts, using
**RAG** over a knowledge base of prompt-engineering technique docs that you control.

Ships with an [evaluation harness](backend/eval/) — because "this makes prompts
better" is a claim, and claims should have numbers behind them.

## 🚀 The Problem

Non-technical users struggle to get good results from LLMs because they don't know
prompt engineering. That means wasted time re-prompting, and wasted tokens.

## 🛠️ Tech Stack

- **Frontend:** Chrome Extension MV3 (vanilla JS/HTML/CSS)
- **Backend:** FastAPI (Python)
- **LLM:** Gemini 3 Flash via LangChain
- **Vector DB:** ChromaDB
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (runs locally, no API cost)

## ✨ Features

**Configurable knowledge base.** The RAG corpus isn't fixed. Add your own technique
docs by pasting text or uploading `.txt`, toggle any doc on or off per query, and
delete what you don't want. Adding a doc chunks and embeds it incrementally — no
full reindex. Managed from the extension's 📚 panel, backed by `/sources`.

**Persona, reasoning, and format controls** for the generated prompt.

**Relevance-gated retrieval.** Chunks are only injected when they clear a similarity
threshold. Below it, the prompt is enhanced with no RAG context at all — which,
per the eval below, matters more than it sounds.

---

## 📊 Evaluation

Full methodology in [`backend/eval/README.md`](backend/eval/README.md). Raw results
in [`backend/eval/results/`](backend/eval/results/).

### How it's measured

Only the **enhancer** varies. The **target** model (which answers the prompt) and the
**judge** are held fixed, so any difference is attributable to the enhancement step
rather than a stronger downstream model.

```
baseline:  lazy prompt ─────────────────────────► target ─► output A
enhanced:  lazy prompt ─► enhancer + RAG ───────► target ─► output B
                                    judge(A, B), then judge(B, A)
```

LLM judges favour whichever response they see first, so every pair is judged twice
with the order swapped. A result counts as a win only if it survives both orderings;
if the two runs disagree it's scored as a tie. Ties stay in the denominator, making
the reported win rate a conservative floor.

The harness was validated before its output was trusted: it correctly picks a strong
answer over a weak one in both slot positions, and returns "tie" on identical inputs.

### Result: the first version made outputs worse

| | v1 — unconditional RAG | v2 — relevance-gated |
|---|---|---|
| Record (W/L/T) | **0 / 10 / 2** | **5 / 2 / 5** |
| Win rate (ties in denominator) | 0.0% | **41.7%** |
| Win rate (decisive only) | 0% | **71%** |
| Token overhead vs. baseline | +226% | **+143%** |
| Token Cost (tokens per win-rate point) | n/a | 948 |

n = 12 prompts, 2 per domain × 6 domains. Target and judge: Gemini 3 Flash.

v1 lost every decisive comparison. That's not sampling noise — at a true 50% win
rate, going 0-for-10 has probability ~0.001.

### Why it failed, and what fixed it

Inspecting the losses showed a specific mechanism. Retrieval always returned `k=2`
chunks regardless of relevance, so **irrelevant context actively hijacked the prompt.**

Given *"write a professional email"*, the retriever matched the **academic writing**
doc. The generated prompt instructed the model to write an email with
**Abstract / Methodology / Results** headings, and the output was an email about an
"ML inference pipeline" the user never mentioned. The un-enhanced baseline just
returned four clean email templates.

Three fixes:

1. **Relevance threshold on retrieval** (`doc_store.MAX_DISTANCE`). Chunks below the
   cutoff are dropped; if nothing clears it, the prompt is built with no context.
   Calibrated against the bundled corpus — on-topic queries scored 0.83–1.74,
   off-topic 1.37–1.65. Those ranges **overlap**, because MiniLM discriminates poorly
   over docs this short, so the cutoff deliberately favours rejecting bad context
   over catching every good hit.
2. **Anti-fabrication rule** in the meta-prompt: underspecified requests must surface
   assumptions or offer options, never invent a scenario.
3. **Proportionate scaffolding** — persona/format/reasoning apply "only where they
   genuinely fit", instead of forcing headings onto tasks that don't want them.

### Objective benchmark: enhancement hurts on already-precise prompts

Win rate is a judge's opinion. GSM8K is ground truth, so it's the harder test.

| | Baseline | Enhanced |
|---|---|---|
| GSM8K accuracy (n=50) | **86.0%** | **78.0%** |

Directionally negative — but **not statistically significant**. The gap is 4 problems
out of 50, and under every discordant-pair split consistent with that net difference,
McNemar's exact test gives p ≥ 0.125. Detecting an effect this size would need several
hundred problems. It should not be quoted as "enhancement reduces accuracy by 8 points."

The *mechanism*, though, is real and reproducible. GSM8K questions are already fully
specified — which is precisely what this tool is not built for. Given:

> Josh buys a house for $80,000 and puts in $50,000 of repairs. This increased the
> value by 150%. How much profit did he make?

the enhancer invented an ambiguity that wasn't in the question, split "increased by
150%" into two competing interpretations, and asked the model to compute **both**.
The anti-fabrication rule that fixed the email case — *surface assumptions rather than
invent context* — backfires here: on a precise problem, the model hedges instead of
committing. For a math question, hedging is a failure even when one branch is right.

Two honest caveats on that number:

- **A grader confound is mixed in.** Accuracy is scored by extracting the last number
  in the output, which misreads multi-scenario answers regardless of reasoning quality.
  How much of the 8 points is hedging versus grader artifact is **unmeasured**. The
  harness now writes `results/gsm8k_items.csv` with per-item outputs and the
  discordant-pair counts a paired test needs — but that was added after this run, so
  separating the two requires re-running the slice.
- **This is off-distribution use.** Applying a vague-prompt tool to fully-specified
  problems tests it outside its intended domain. That is worth knowing — it defines
  where the tool should and shouldn't fire — but it is not evidence about its
  performance on the prompts it targets.

**The actionable conclusion:** enhancement should detect prompt specificity and skip
itself when the input is already well-formed. That's the next thing to build.

### Honest limitations

- **n = 12 is small.** The 95% CI on a win rate at this size is roughly ±25 points.
  Treat these as directional; run the full 60-prompt set before quoting them hard.
- **The `general` domain still shows 0%**, and GSM8K trends negative. Together these
  mark the boundary: enhancement earns its cost on open-ended, under-specified
  requests, not on simple or already-precise ones.
- **Judge and target are the same model family**, which risks shared blind spots.
  Both conditions face the same judge, so the comparison is fair, but a
  cross-family judge would be a stronger design.
- **Enhancement is not free.** It still costs ~143% more tokens than answering
  directly. The case for it rests on output quality, not token savings.
- **No re-prompting data.** Claims about "prompts avoided" or "time saved" would need
  a user study measuring turns-to-satisfaction. Not run, so not claimed.

### Positioning the token cost

Enhancement costs **2.43×** the tokens of answering directly. For context, every
quality-improving prompting technique costs tokens:

| Technique | Token multiplier |
|---|---|
| Direct answer | 1× |
| **This enhancer** | **2.43×** |
| Zero-shot CoT | [2–3×](https://tianpan.co/blog/2026-04-10-token-economics-chain-of-thought-when-thinking-costs-more) |
| Self-Consistency | [5–20×](https://arxiv.org/html/2511.00751v2) |
| Tree-of-Thoughts | 10–50× |

For comparison, self-consistency on Gemini 2.5 buys +0.4% on HotpotQA at ~20× tokens
and +1.6% on MATH-500 at ~15× — which [its own authors call](https://arxiv.org/html/2511.00751v2)
"difficult to justify." This tool sits at the cheap end of that range.

The claim to avoid is "saves tokens" — it does not; it costs 143% more. The claim the
data supports is **quality per token spent**, measured by Token Cost
([arXiv:2505.14880](https://arxiv.org/html/2505.14880v1)): 948 tokens per win-rate point.

Note that win rate and the accuracy figures in those papers are different units, so
this is a cost comparison, not a head-to-head quality one.

### Reproducing

```bash
cd backend/eval
python run_eval.py --enhancers gemini --per-domain 2   # the judged run above
python run_eval.py --enhancers gemini --skip-pairwise --gsm8k 50   # the GSM8K run
python run_eval.py --enhancers gemini                  # full 60-prompt set
```

Metric definitions, including Token Cost (adapted from
[arXiv:2505.14880](https://arxiv.org/html/2505.14880v1)), are in the eval README.

---

## 🛠️ Local Setup

```bash
git clone https://github.com/h1tenM/prompt-enhancer-ext.git
cd prompt-enhancer-ext/backend

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create `backend/.env`:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

Then run the server:

```bash
python main.py
```

Load the extension: `chrome://extensions` → enable Developer mode → **Load unpacked**
→ select the repo root.

The vector DB seeds itself from `backend/research_docs/` on first run. To rebuild it
from scratch (this wipes docs added through the UI):

```bash
python ingest.py
```

## 🔭 What's next

- **Specificity gating** — skip enhancement when the input is already well-formed.
  This is the clearest finding from the eval and the highest-value fix.
- Re-run the GSM8K slice (the harness now persists per-item data) and re-grade it to
  separate genuine hedging from the last-number grader artifact
- Serverless backend so it works without a local server
- Cross-family judge, and the full 60-prompt run across multiple enhancer models
- Better embeddings — the threshold overlap above is the current ceiling on retrieval
  precision
- A user study to measure re-prompting, the one claim the current harness can't support
