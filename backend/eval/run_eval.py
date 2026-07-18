"""Measures whether prompt enhancement actually helps, and what it costs.

Experimental design
-------------------
Only the ENHANCER model varies. The TARGET model (which answers the prompt) and
the JUDGE are held fixed across every condition, so a difference in win rate is
attributable to the enhancer rather than to a stronger downstream model.

  baseline:  lazy prompt ------------------------> target -> output A
  enhanced:  lazy prompt -> enhancer + RAG ------> target -> output B
             judge(A, B) twice, order swapped

Metrics
-------
  win_rate      share of prompts where enhanced beat baseline (ties excluded
                from the numerator, kept in the denominator)
  overhead      extra tokens the enhancement step costs
  token_cost    total tokens / win-rate percentage point, adapted from
                "Incorporating Token Usage into Prompting Strategy Evaluation"
                (arXiv:2505.14880). Lower is better.
  gsm8k         exact-match accuracy, lazy vs enhanced
  humaneval     pass@1, lazy vs enhanced (opt-in, executes generated code)

Usage
-----
  python run_eval.py --enhancers gemini llama-8b qwen-32b
  python run_eval.py --enhancers gemini --limit 10        # quick smoke run
  python run_eval.py --enhancers gemini --gsm8k 50
  python run_eval.py --enhancers gemini --humaneval 20 --enable-code-exec
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import benchmarks  # noqa: E402
import providers  # noqa: E402
from judge import judge_pair  # noqa: E402

from doc_store import DocStore  # noqa: E402
from prompt_builder import build_meta_prompt  # noqa: E402
from langchain_chroma import Chroma  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(EVAL_DIR)
RESULTS_DIR = os.path.join(EVAL_DIR, "results")

def build_retriever():
    """Same embeddings + corpus the live extension uses."""
    db_dir = os.path.join(BACKEND_DIR, "db")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    return DocStore(
        vectorstore,
        os.path.join(db_dir, "docs.json"),
        seed_dir=os.path.join(BACKEND_DIR, "research_docs"),
    )


def enhance(enhancer, store, prompt):
    """Mirrors /enhance exactly — same retriever, same threshold, same template."""
    docs = store.search(prompt, k=2)
    meta = build_meta_prompt(
        prompt,
        persona="Expert",
        reasoning="Think step-by-step",
        format="Markdown with clear headings",
        context="\n".join(d.page_content for d in docs),
    )
    return enhancer.complete(meta)


# --- pairwise quality --------------------------------------------------


def run_pairwise(enhancer, target, judge, store, prompts, out_rows):
    tally = defaultdict(int)
    by_domain = defaultdict(lambda: defaultdict(int))
    overhead_tokens = 0
    baseline_tokens = 0
    enhanced_tokens = 0

    for i, item in enumerate(prompts, 1):
        print(f"  [{i}/{len(prompts)}] {item['id']}: {item['text'][:48]}…", flush=True)
        try:
            baseline = target.complete(item["text"])
            enhancement = enhance(enhancer, store, item["text"])
            enhanced = target.complete(enhancement.text)
            winner, _ = judge_pair(judge, item["text"], baseline.text, enhanced.text)
        except Exception as e:
            print(f"      skipped ({type(e).__name__}: {e})")
            continue

        tally[winner] += 1
        tally["total"] += 1
        by_domain[item["domain"]][winner] += 1
        by_domain[item["domain"]]["total"] += 1

        overhead_tokens += enhancement.total_tokens
        baseline_tokens += baseline.total_tokens
        enhanced_tokens += enhanced.total_tokens

        out_rows.append(
            {
                "enhancer": enhancer.name,
                "prompt_id": item["id"],
                "domain": item["domain"],
                "winner": winner,
                "enhancement_tokens": enhancement.total_tokens,
                "baseline_tokens": baseline.total_tokens,
                "enhanced_tokens": enhanced.total_tokens,
            }
        )
        time.sleep(0.5)  # stay inside free-tier rate limits

    return tally, by_domain, overhead_tokens, baseline_tokens, enhanced_tokens


# --- objective benchmarks ----------------------------------------------


def run_gsm8k(enhancer, target, store, n):
    """Per-item outcomes are persisted to gsm8k_items.csv.

    Two reasons. First, the aggregate can't be re-graded without them, and the
    default grader (last number in the output) misreads answers that present
    more than one scenario — a real confound, since enhancement makes the model
    hedge on already-precise problems. Second, a paired test like McNemar's
    needs the discordant pairs, which the totals alone throw away.
    """
    tasks = benchmarks.load_gsm8k(n)
    lazy_correct = enhanced_correct = 0
    both = neither = only_lazy = only_enhanced = 0
    items = []

    for i, task in enumerate(tasks, 1):
        print(f"  [{i}/{len(tasks)}] {task['id']}", flush=True)
        try:
            lazy = target.complete(task["question"])
            enhancement = enhance(enhancer, store, task["question"])
            enhanced = target.complete(enhancement.text)
        except Exception as e:
            print(f"      skipped ({type(e).__name__})")
            continue

        lazy_ok = benchmarks.grade_gsm8k(lazy.text, task["answer"])
        enhanced_ok = benchmarks.grade_gsm8k(enhanced.text, task["answer"])
        lazy_correct += lazy_ok
        enhanced_correct += enhanced_ok

        both += lazy_ok and enhanced_ok
        neither += not lazy_ok and not enhanced_ok
        only_lazy += lazy_ok and not enhanced_ok
        only_enhanced += enhanced_ok and not lazy_ok

        items.append(
            {
                "id": task["id"],
                "gold": task["answer"],
                "baseline_correct": int(lazy_ok),
                "enhanced_correct": int(enhanced_ok),
                "baseline_output": lazy.text.replace("\n", " ")[-400:],
                "enhanced_output": enhanced.text.replace("\n", " ")[-400:],
                "enhanced_prompt": enhancement.text.replace("\n", " ")[:400],
            }
        )
        time.sleep(0.5)

    if items:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(os.path.join(RESULTS_DIR, "gsm8k_items.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(items[0]))
            writer.writeheader()
            writer.writerows(items)

    total = len(tasks)
    return {
        "n": total,
        "baseline_accuracy": round(100 * lazy_correct / total, 1) if total else None,
        "enhanced_accuracy": round(100 * enhanced_correct / total, 1) if total else None,
        # Discordant pairs — the only cells McNemar's test actually uses.
        "paired": {
            "both_correct": both,
            "neither_correct": neither,
            "only_baseline_correct": only_lazy,
            "only_enhanced_correct": only_enhanced,
        },
    }


def run_humaneval(enhancer, target, store, n):
    tasks = benchmarks.load_humaneval(n)
    lazy_pass = enhanced_pass = 0

    for i, task in enumerate(tasks, 1):
        print(f"  [{i}/{len(tasks)}] {task['id']}", flush=True)
        request = f"Complete this Python function:\n\n{task['prompt']}"
        try:
            lazy = target.complete(request)
            enhancement = enhance(enhancer, store, request)
            enhanced = target.complete(enhancement.text)
        except Exception as e:
            print(f"      skipped ({type(e).__name__})")
            continue

        lazy_pass += benchmarks.grade_humaneval(lazy.text, task)
        enhanced_pass += benchmarks.grade_humaneval(enhanced.text, task)
        time.sleep(0.5)

    total = len(tasks)
    return {
        "n": total,
        "baseline_pass_at_1": round(100 * lazy_pass / total, 1) if total else None,
        "enhanced_pass_at_1": round(100 * enhanced_pass / total, 1) if total else None,
    }


# --- reporting ---------------------------------------------------------


def summarize(tally, by_domain, overhead, baseline_tok, enhanced_tok):
    total = tally["total"]
    if not total:
        return None

    win_rate = round(100 * tally["enhanced"] / total, 1)
    total_tokens = overhead + enhanced_tok

    return {
        "n_prompts": total,
        "wins": tally["enhanced"],
        "losses": tally["baseline"],
        "ties": tally["tie"],
        "win_rate_pct": win_rate,
        "avg_overhead_tokens": round(overhead / total),
        "avg_baseline_tokens": round(baseline_tok / total),
        "avg_enhanced_tokens": round(enhanced_tok / total),
        "token_overhead_pct": round(100 * (total_tokens - baseline_tok) / baseline_tok, 1)
        if baseline_tok
        else None,
        # Token Cost (arXiv:2505.14880): tokens spent per point of win rate.
        "token_cost": round(total_tokens / win_rate, 1) if win_rate else None,
        "by_domain": {
            d: {
                "n": c["total"],
                "win_rate_pct": round(100 * c["enhanced"] / c["total"], 1),
            }
            for d, c in sorted(by_domain.items())
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enhancers", nargs="+", default=["gemini"])
    parser.add_argument("--target", default="gemini")
    parser.add_argument("--judge", default="gemini")
    parser.add_argument("--limit", type=int, help="cap the pairwise prompt count")
    parser.add_argument(
        "--per-domain",
        type=int,
        help="take N prompts from each domain instead of the first N overall",
    )
    parser.add_argument("--gsm8k", type=int, default=0, help="GSM8K problems to run")
    parser.add_argument("--humaneval", type=int, default=0)
    parser.add_argument(
        "--enable-code-exec",
        action="store_true",
        help="required for --humaneval: runs model-generated code locally",
    )
    parser.add_argument(
        "--skip-pairwise",
        action="store_true",
        help="run only the objective benchmarks, merging into the existing report "
        "instead of discarding its pairwise numbers",
    )
    args = parser.parse_args()

    if args.humaneval and not args.enable_code_exec:
        parser.error(
            "--humaneval executes model-generated Python on this machine. "
            "Pass --enable-code-exec to acknowledge, or drop --humaneval."
        )

    with open(os.path.join(EVAL_DIR, "prompts.json")) as f:
        prompts = json.load(f)["prompts"]
    if args.per_domain:
        seen = defaultdict(int)
        kept = []
        for p in prompts:
            if seen[p["domain"]] < args.per_domain:
                seen[p["domain"]] += 1
                kept.append(p)
        prompts = kept
    if args.limit:
        prompts = prompts[: args.limit]

    print("Loading retriever…")
    store = build_retriever()
    target = providers.build(args.target)
    judge = providers.build(args.judge)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, "report.json")

    # --skip-pairwise adds benchmark numbers to an existing run rather than
    # replacing it, so a cheap benchmark pass can't wipe an expensive judged one.
    prior = {}
    if args.skip_pairwise and os.path.exists(report_path):
        with open(report_path) as f:
            prior = json.load(f).get("enhancers", {})

    report = {
        "target_model": target.name,
        "judge_model": judge.name,
        "enhancers": {},
    }
    rows = []

    for key in args.enhancers:
        print(f"\n=== enhancer: {key} ===")
        try:
            enhancer = providers.build(key)
        except (KeyError, RuntimeError) as e:
            print(f"  unavailable: {e}")
            continue

        if args.skip_pairwise:
            summary = dict(prior.get(enhancer.name, {}))
            if summary:
                print(f"  reusing pairwise result ({summary['win_rate_pct']}% win rate)")
            else:
                print("  no prior pairwise result to merge into")
        else:
            tally, by_domain, overhead, base_tok, enh_tok = run_pairwise(
                enhancer, target, judge, store, prompts, rows
            )
            summary = summarize(tally, by_domain, overhead, base_tok, enh_tok)
            if not summary:
                print("  no successful comparisons")
                continue

        if args.gsm8k:
            print(f"\n  --- GSM8K ({args.gsm8k}) ---")
            summary["gsm8k"] = run_gsm8k(enhancer, target, store, args.gsm8k)
        if args.humaneval:
            print(f"\n  --- HumanEval ({args.humaneval}) ---")
            summary["humaneval"] = run_humaneval(enhancer, target, store, args.humaneval)

        report["enhancers"][enhancer.name] = summary
        if "win_rate_pct" in summary:
            print(
                f"\n  win rate {summary['win_rate_pct']}%  "
                f"({summary['wins']}W/{summary['losses']}L/{summary['ties']}T)  "
                f"overhead +{summary['token_overhead_pct']}%  "
                f"token cost {summary['token_cost']}"
            )
        if "gsm8k" in summary:
            g = summary["gsm8k"]
            print(
                f"  GSM8K (n={g['n']}): baseline {g['baseline_accuracy']}%  "
                f"-> enhanced {g['enhanced_accuracy']}%"
            )

    if not report["enhancers"]:
        print("\nNo enhancer produced results.")
        return

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    if rows:
        with open(os.path.join(RESULTS_DIR, "per_prompt.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    # Feed the extension's dashboard from the best measured configuration.
    # Only judged runs qualify — a benchmark-only pass has no win rate to report.
    judged = {k: v for k, v in report["enhancers"].items() if "win_rate_pct" in v}
    if judged:
        best_name, best = max(judged.items(), key=lambda kv: kv[1]["win_rate_pct"])
        with open(os.path.join(RESULTS_DIR, "metrics_config.json"), "w") as f:
            json.dump(
                {
                    "win_rate_pct": best["win_rate_pct"],
                    "n_prompts": best["n_prompts"],
                    "enhancer_model": best_name,
                    "judge_model": report["judge_model"],
                    "measured_at": time.strftime("%Y-%m-%d"),
                },
                f,
                indent=2,
            )
        print(f"\nWrote results to {RESULTS_DIR}/ (best enhancer: {best_name})")
    else:
        print(f"\nWrote results to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
