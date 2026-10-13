"""Follow-up questions: rewriting a question to stand on its own, using the conversation so far.

Component 8D (conversation memory).

"And for headers?" finds nothing by itself. Before retrieval, the model rewrites a follow-up
into a standalone question ("How do I declare header parameters?") from the last few turns.
That question is then searched and answered exactly like a first question, with the same
answer prompt: earlier turns help to understand the question, they are never a source.
"""

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import structlog

from app.rag.llm import LLMProvider, LLMResult

logger = structlog.get_logger(__name__)

CONDENSE_PROMPT_VERSION = "condense_v1"
PROMPTS_DIR = Path(__file__).parent / "prompts"
MAX_TURNS = 3  # earlier exchanges given to the rewrite
ANSWER_PREVIEW_CHARS = 400  # of each earlier answer: enough to resolve "it", short to read
MAX_QUERY_CHARS = 1000  # same limit as a question
QUERY_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
}


@dataclass(frozen=True)
class Turn:
    question: str
    answer: str | None  # None when the system abstained or the turn failed


@dataclass(frozen=True)
class Condensed:
    query: str  # what to search for and answer
    rewritten: bool  # False: the question was used as it is
    llm_call: LLMResult | None = None


@cache
def load_condense_prompt() -> str:
    return (PROMPTS_DIR / f"{CONDENSE_PROMPT_VERSION}.txt").read_text(encoding="utf-8").strip()


def build_condense_prompt(history: list[Turn], question: str) -> tuple[str, str]:
    lines = ["Conversation so far:"]
    for turn in history[-MAX_TURNS:]:
        lines.append(f"<user>{turn.question}</user>")
        answer = (turn.answer or "(no answer found)")[:ANSWER_PREVIEW_CHARS]
        lines.append(f"<assistant>{answer}</assistant>")
    lines.append(f"\nLast question: {question}")
    return load_condense_prompt(), "\n".join(lines)


def condense(llm: LLMProvider, history: list[Turn], question: str) -> Condensed:
    """The question to search for. Without history, the question itself (no model call).

    If the model's reply can't be used, the original question is used: a follow-up then finds
    less, but the request doesn't fail over a helper step.
    """
    if not history:
        return Condensed(query=question, rewritten=False)
    system, user = build_condense_prompt(history, question)
    result = llm.generate(system, user, max_tokens=120, json_schema=QUERY_SCHEMA)
    try:
        query = str(json.loads(result.text)["query"]).strip()
    except (json.JSONDecodeError, KeyError, TypeError):
        query = ""
    if not query or len(query) > MAX_QUERY_CHARS:
        logger.warning("condense_failed", chars=len(result.text))
        return Condensed(query=question, rewritten=False, llm_call=result)
    return Condensed(query=query, rewritten=query != question, llm_call=result)
