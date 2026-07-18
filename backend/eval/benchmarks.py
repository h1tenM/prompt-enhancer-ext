"""Objective benchmark slices — accuracy you can check without a judge.

These carry more weight than win rate because correctness is ground-truth, not
opinion. GSM8K is graded by exact match on the final number. HumanEval requires
executing model-written code and is therefore opt-in; see run_humaneval below.
"""

import gzip
import io
import json
import os
import re
import subprocess
import sys
import tempfile

import httpx

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")

GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/test.jsonl"
)
HUMANEVAL_URL = (
    "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
)


def _cached(filename, url, gz=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(path):
        print(f"  downloading {filename}…")
        response = httpx.get(url, timeout=120.0, follow_redirects=True)
        response.raise_for_status()
        raw = response.content
        if gz:
            raw = gzip.decompress(raw)
        with open(path, "wb") as f:
            f.write(raw)
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# --- GSM8K -------------------------------------------------------------

_NUMBER = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def _final_number(text):
    matches = _NUMBER.findall(text or "")
    if not matches:
        return None
    cleaned = matches[-1].replace(",", "").replace("$", "").rstrip(".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def load_gsm8k(n):
    rows = _cached("gsm8k_test.jsonl", GSM8K_URL)[:n]
    return [
        {
            "id": f"gsm8k-{i:03d}",
            "question": r["question"],
            "answer": float(r["answer"].split("####")[-1].strip().replace(",", "")),
        }
        for i, r in enumerate(rows)
    ]


def grade_gsm8k(output, gold):
    predicted = _final_number(output)
    return predicted is not None and abs(predicted - gold) < 1e-4


# --- HumanEval ---------------------------------------------------------

_CODE_BLOCK = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def load_humaneval(n):
    rows = _cached("humaneval.jsonl", HUMANEVAL_URL, gz=True)[:n]
    return [
        {
            "id": r["task_id"],
            "prompt": r["prompt"],
            "test": r["test"],
            "entry_point": r["entry_point"],
        }
        for r in rows
    ]


def _extract_code(output, fallback_prompt):
    blocks = _CODE_BLOCK.findall(output or "")
    if blocks:
        return max(blocks, key=len)
    # No fence — assume the model continued the function signature directly.
    return fallback_prompt + (output or "")


def grade_humaneval(output, task, timeout=10):
    """Execute the generated solution against HumanEval's unit tests.

    Runs in a separate interpreter with a hard timeout and a throwaway cwd. This
    is still arbitrary model-written code executing on your machine — the harness
    keeps it behind --enable-code-exec for that reason.
    """
    code = _extract_code(output, task["prompt"])
    program = f"{code}\n\n{task['test']}\n\ncheck({task['entry_point']})\n"

    with tempfile.TemporaryDirectory() as workdir:
        script = os.path.join(workdir, "candidate.py")
        with open(script, "w") as f:
            f.write(program)
        try:
            result = subprocess.run(
                [sys.executable, script],
                cwd=workdir,
                capture_output=True,
                timeout=timeout,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except OSError:
            return False
