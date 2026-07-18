# Evaluation Harness

Measures whether prompt enhancement actually improves output quality, and what it
costs in tokens. Built because the alternative — asserting a benefit — isn't
something you can defend when someone asks how you measured it.

## Design

Only the **enhancer** varies. The **target** model (which answers the prompt) and
the **judge** are fixed across every condition, so any difference in win rate is
attributable to the enhancer rather than to a stronger downstream model.

```
baseline:  lazy prompt ─────────────────────────► target ─► output A
enhanced:  lazy prompt ─► enhancer + RAG ───────► target ─► output B
                                    judge(A, B), then judge(B, A)
```

**Position-bias control.** LLM judges systematically favour whichever response
they see first, and the effect persists across model families. Every pair is
therefore judged twice with the order swapped. A result only counts as a win if
it survives both orderings; if the two runs disagree, it's recorded as a tie —
the honest reading, since the judge had no stable preference.

## Metrics

| Metric | Definition |
|---|---|
| `win_rate_pct` | Share of prompts where enhanced beat baseline. Ties stay in the denominator, so this is a conservative floor. |
| `token_overhead_pct` | Extra tokens the enhancement step costs vs. answering the lazy prompt directly. |
| `token_cost` | Total tokens ÷ win-rate percentage point. Lower is better. Adapted from [arXiv:2505.14880](https://arxiv.org/html/2505.14880v1), which defines TC as tokens per unit of accuracy. |
| `gsm8k` | Exact-match accuracy on GSM8K, lazy vs. enhanced. Ground truth, no judge involved. |
| `humaneval` | pass@1 on HumanEval, lazy vs. enhanced. Opt-in — see the warning below. |

Win rate answers "is it better?"; token cost answers "was it worth it?". Reporting
either alone is how you end up with a claim that doesn't survive scrutiny.

## Running it

```bash
cd backend/eval

python run_eval.py --enhancers gemini                      # full 60-prompt set
python run_eval.py --enhancers gemini --per-domain 2       # quick stratified pass
python run_eval.py --enhancers gemini llama-8b qwen-32b    # model comparison
python run_eval.py --enhancers gemini --gsm8k 50           # + objective benchmark

# Benchmarks only, merged into the existing report rather than replacing it.
# Use this so a cheap benchmark pass can't discard an expensive judged run.
python run_eval.py --enhancers gemini --skip-pairwise --gsm8k 50
```

Comparing against Groq-hosted models needs `GROQ_API_KEY` in `backend/.env`
(free at [console.groq.com](https://console.groq.com)). Available keys are in
`providers.REGISTRY`; Groq's free-tier lineup changes, so check their model list
if one starts 404ing.

### HumanEval executes model-written code

`--humaneval` runs code the model generated, on your machine. The harness isolates
it in a subprocess with a throwaway working directory and a hard timeout, but that
is mitigation, not a sandbox. It's gated behind `--enable-code-exec` so it can
never run by accident:

```bash
python run_eval.py --enhancers gemini --humaneval 20 --enable-code-exec
```

If you're at all unsure, use `--gsm8k` instead — it grades by string comparison
and executes nothing.

## Output

Written to `results/`:

- `report.json` — full summary, overall and per-domain
- `per_prompt.csv` — one row per comparison, for your own slicing
- `gsm8k_items.csv` — per-item GSM8K outcomes and outputs
- `metrics_config.json` — the win rate the extension's dashboard reads

### A known grading caveat

GSM8K is graded by extracting the **last number** in the output. That heuristic
misreads answers presenting more than one scenario — and enhancement makes the model
hedge exactly that way on already-precise problems, so some of the measured accuracy
drop is grader artifact rather than reasoning failure. `gsm8k_items.csv` keeps the raw
outputs so a stricter grader can be applied without re-running the model calls.

`report.json` also records the discordant-pair counts (`only_baseline_correct` /
`only_enhanced_correct`), which are the cells McNemar's exact test needs. Accuracy
totals alone can't support a paired significance test.

`metrics_config.json` is the only link back into the app. If the eval has never
been run, the dashboard shows `—` rather than inventing a number.

## Cost

Each pairwise comparison is 5 model calls (baseline, enhance, enhanced, judge ×2).
The full 60-prompt set is ~300 calls per enhancer, which fits in free-tier quotas
but will take a while — there's a deliberate 0.5s pause between items to stay
inside rate limits.

## Interpreting small runs

`--per-domain 2` is 12 prompts. At that size the 95% confidence interval on a win
rate is roughly ±25 points, so it tells you whether the pipeline works, not
whether enhancement helps. Use the full set before quoting a number anywhere.
