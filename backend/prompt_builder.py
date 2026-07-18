"""The meta-prompt, shared by the live API and the eval harness.

Kept in one place deliberately: if the eval measured a different prompt than the
app ships, its numbers would be worthless.

Rules 1-4 exist because of measured failures. The v1 template scored a 0% win
rate over 12 prompts, and inspection showed why: given "write a professional
email" it invented an ML-pipeline scenario and imposed Abstract/Methodology/
Results headings on an email. The rules target invention and over-scaffolding
specifically.
"""

TEMPLATE = """You are a Master Prompt Engineer. Rewrite a vague user request into a clear, effective prompt.

USER CONFIGURATION — apply each only where it genuinely fits the request:
- PERSONA: {persona}
- REASONING MODE: {reasoning}
- OUTPUT FORMAT: {format}
{context_block}
RULES:
1. Preserve the user's actual intent. Do not narrow a broad request into one specific scenario.
2. Never invent facts, topics, names, or context the user did not supply. Where the request is
   underspecified, have the prompt ask the model to state its assumptions or offer a few options —
   not to fabricate a situation.
3. Keep the prompt proportionate. A simple request should produce a short prompt.
4. Do not impose headings, sections, or formatting that would be inappropriate for the task.

USER REQUEST: "{user_prompt}"

Return ONLY the optimized prompt, with no preamble or commentary."""

CONTEXT_BLOCK = """
RELEVANT TECHNIQUES — apply only those that suit this request:
{context}
"""


def build_meta_prompt(prompt, persona, reasoning, format, context=""):
    """Assemble the meta-prompt. Omits the techniques section entirely when
    retrieval found nothing relevant, rather than padding it with a placeholder
    the model might try to satisfy."""
    return TEMPLATE.format(
        persona=persona,
        reasoning=reasoning,
        format=format,
        context_block=CONTEXT_BLOCK.format(context=context) if context else "",
        user_prompt=prompt,
    )
