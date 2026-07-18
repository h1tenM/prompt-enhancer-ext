"""Uniform completion interface across providers, so the harness can swap the
enhancer model without touching the experiment code.

Every provider returns a Completion carrying real token counts from the API
response — the eval's cost numbers are only as honest as these are, so we never
estimate from string length here.
"""

import os
from dataclasses import dataclass

import httpx

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TIMEOUT = 120.0


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens


class GeminiProvider:
    """Google AI Studio free tier."""

    def __init__(self, model="gemini-3-flash-preview", temperature=0.3):
        from langchain_google_genai import ChatGoogleGenerativeAI

        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY not set")

        self.name = f"gemini/{model}"
        self.llm = ChatGoogleGenerativeAI(
            model=model, google_api_key=key, temperature=temperature
        )

    def complete(self, prompt):
        response = self.llm.invoke(prompt)
        content = response.content
        if isinstance(content, list):
            content = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in content
            )
        usage = getattr(response, "usage_metadata", None) or {}
        return Completion(
            text=content,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )


class GroqProvider:
    """Groq free tier — OpenAI-compatible, so plain httpx avoids another SDK."""

    def __init__(self, model, temperature=0.3):
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY not set (get one at console.groq.com)")

        self.name = f"groq/{model}"
        self.model = model
        self.temperature = temperature
        self.headers = {"Authorization": f"Bearer {key}"}

    def complete(self, prompt):
        response = httpx.post(
            GROQ_URL,
            headers=self.headers,
            timeout=GROQ_TIMEOUT,
            json={
                "model": self.model,
                "temperature": self.temperature,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage", {})
        return Completion(
            text=data["choices"][0]["message"]["content"],
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )


# Models the harness knows how to build. Groq's free-tier lineup shifts over
# time; check console.groq.com/docs/models if one starts 404ing.
REGISTRY = {
    "gemini": lambda: GeminiProvider("gemini-3-flash-preview"),
    "llama-8b": lambda: GroqProvider("llama-3.1-8b-instant"),
    "qwen-32b": lambda: GroqProvider("qwen3-32b"),
    "llama-70b": lambda: GroqProvider("llama-3.3-70b-versatile"),
}


def build(key):
    if key not in REGISTRY:
        raise KeyError(f"Unknown model '{key}'. Options: {', '.join(REGISTRY)}")
    return REGISTRY[key]()
