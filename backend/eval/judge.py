"""Pairwise LLM-as-judge with position-bias control.

Judges systematically favour whichever response they see first (the effect holds
across model families), so every pair is judged twice with the order swapped.
A verdict only counts as a win if it survives both orderings; disagreement
between the two runs is recorded as a tie, which is the honest reading — the
judge had no stable preference.
"""

import re

JUDGE_TEMPLATE = """You are evaluating two AI responses to the same underlying user need.

USER'S UNDERLYING NEED: "{need}"

--- RESPONSE A ---
{response_a}

--- RESPONSE B ---
{response_b}

Judge which response better serves the user's need, weighing:
1. Specificity and actionability — does it give concrete, usable detail?
2. Structure — is it organised so the user can act on it?
3. Completeness — does it anticipate what the user actually needed?
4. Appropriate assumptions — does it handle the vagueness of the request well?

Ignore length unless extra length adds real substance. Ignore formatting flourish.

Answer with exactly one line:
VERDICT: A
or
VERDICT: B
or
VERDICT: TIE
"""

_VERDICT = re.compile(r"VERDICT:\s*(A|B|TIE)", re.IGNORECASE)


def _parse(text):
    match = _VERDICT.search(text or "")
    return match.group(1).upper() if match else "TIE"


def judge_pair(judge, need, baseline_output, enhanced_output):
    """Return ("enhanced" | "baseline" | "tie", tokens_used).

    Run 1 shows baseline first; run 2 swaps them. The two runs must agree on the
    same underlying answer for it to count as a decisive result.
    """
    tokens = 0

    first = judge.complete(
        JUDGE_TEMPLATE.format(
            need=need, response_a=baseline_output, response_b=enhanced_output
        )
    )
    tokens += first.total_tokens
    verdict_1 = _parse(first.text)

    second = judge.complete(
        JUDGE_TEMPLATE.format(
            need=need, response_a=enhanced_output, response_b=baseline_output
        )
    )
    tokens += second.total_tokens
    verdict_2 = _parse(second.text)

    # Map each run's A/B onto which system it actually referred to.
    winner_1 = {"A": "baseline", "B": "enhanced", "TIE": "tie"}[verdict_1]
    winner_2 = {"A": "enhanced", "B": "baseline", "TIE": "tie"}[verdict_2]

    if winner_1 == winner_2:
        return winner_1, tokens
    return "tie", tokens
