"""LLM-as-judge for faithfulness and correctness.

Component 4D.

The judge should be a different model family from the one that writes the answers, so it is not
grading its own style of output (self-preference bias). Its prompts are versioned like the answer
prompt, and every results file records the judge model and prompt versions.
"""

import json
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

from app.rag.llm import LLMProvider

PROMPTS_DIR = Path(__file__).parent / "prompts"
FAITHFULNESS_PROMPT = "judge_faithfulness_v1"
CORRECTNESS_PROMPT = "judge_correctness_v1"
JUDGE_MAX_TOKENS = 600

_CITATION = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")

FAITHFULNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"claim": {"type": "string"}, "supported": {"type": "boolean"}},
                "required": ["claim", "supported"],
            },
        }
    },
    "required": ["claims"],
}
CORRECTNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "partial", "incorrect"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
}
VERDICT_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


class JudgeError(Exception):
    """The judge's reply could not be parsed."""


@dataclass(frozen=True)
class Claim:
    claim: str
    supported: bool


@dataclass(frozen=True)
class Faithfulness:
    score: float | None  # supported claims / all claims; None if the answer made no claims
    claims: list[Claim]


@dataclass(frozen=True)
class Correctness:
    verdict: Literal["correct", "partial", "incorrect"]
    score: float
    reason: str


@cache
def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8").strip()


class Judge:
    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    @property
    def model(self) -> str:
        return self.llm.model

    def faithfulness(self, answer: str, sources: list[str]) -> Faithfulness:
        numbered = "\n\n".join(f"[{n}] {text}" for n, text in enumerate(sources, start=1))
        user = f"Sources:\n\n{numbered}\n\nAnswer:\n{strip_citations(answer)}"
        data = self._ask(FAITHFULNESS_PROMPT, user, FAITHFULNESS_SCHEMA)
        try:
            claims = [Claim(str(c["claim"]), bool(c["supported"])) for c in data["claims"]]
        except (KeyError, TypeError) as exc:
            raise JudgeError(f"Unexpected faithfulness reply: {data!r}") from exc
        score = sum(c.supported for c in claims) / len(claims) if claims else None
        return Faithfulness(score=score, claims=claims)

    def correctness(self, question: str, reference: str, answer: str) -> Correctness:
        user = (
            f"Question: {question}\n\nReference answer: {reference}\n\n"
            f"Answer to grade: {strip_citations(answer)}"
        )
        data = self._ask(CORRECTNESS_PROMPT, user, CORRECTNESS_SCHEMA)
        verdict = data.get("verdict") if isinstance(data, dict) else None
        if verdict not in VERDICT_SCORE:
            raise JudgeError(f"Unexpected correctness reply: {data!r}")
        return Correctness(verdict, VERDICT_SCORE[verdict], str(data.get("reason", "")))

    def _ask(self, prompt: str, user: str, schema: dict) -> dict:
        result = self.llm.generate(
            load_prompt(prompt), user, max_tokens=JUDGE_MAX_TOKENS, json_schema=schema
        )
        try:
            return json.loads(result.text)
        except json.JSONDecodeError as exc:
            raise JudgeError(f"Judge reply is not JSON: {result.text[:200]!r}") from exc


def strip_citations(answer: str) -> str:
    """The judge grades the claims, not the [n] markers."""
    return _CITATION.sub("", answer).strip()


def get_judge() -> Judge:
    """A judge on the configured judge model, served by the same OpenAI-compatible endpoint."""
    from app.core.config import get_settings
    from app.rag.llm import OpenAICompatibleProvider

    settings = get_settings()
    return Judge(
        OpenAICompatibleProvider(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.judge_model,
            context_tokens=settings.llm_context_tokens,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
    )
