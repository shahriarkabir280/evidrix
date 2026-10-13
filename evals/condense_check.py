"""Checks the follow-up rewrite on six cases with the real model.

Component 8D.

    uv run python -m evals.condense_check condense_v2

Five follow-ups that depend on earlier turns and one question that stands on its own; a
rewrite passes when it contains the words that make it standalone. A smoke test of the prompt
with the configured model, not a benchmark: run it after changing either.
"""

import sys

import app.rag.conversation as conv
from app.rag.conversation import Turn, condense
from app.rag.llm import get_llm

conv.CONDENSE_PROMPT_VERSION = sys.argv[1]
conv.load_condense_prompt.cache_clear()
COOKIE = Turn(
    "How do I declare a cookie parameter in FastAPI?", "Use Annotated[str | None, Cookie()] [1]."
)
HEADER = Turn(
    "How do I declare a header parameter in FastAPI?",
    "Use Header; underscores are converted to hyphens [1].",
)
RAGAS = Turn(
    "What does the Ragas paper propose?",
    "A reference-free evaluation framework for RAG pipelines [1].",
)
SORT = Turn(
    "How do I sort a list of tuples by the second item?", "sorted(data, key=itemgetter(1)) [1]."
)
cases = [
    ([COOKIE], "And for headers?", ["header"]),
    ([COOKIE, HEADER], "What happens to underscores in their names?", ["header", "underscore"]),
    ([COOKIE, HEADER], "Can I turn that conversion off?", ["underscore", "header"]),
    ([RAGAS], "Which metrics does it use?", ["ragas", "metric"]),
    ([SORT], "And in reverse order?", ["sort", "reverse"]),
    (
        [SORT],
        "How does the logging module rotate files?",
        ["logging", "rotate"],
    ),  # standalone already
]
llm, ok = get_llm(), 0
for history, question, needed in cases:
    query = condense(llm, history, question).query
    good = all(word in query.lower() for word in needed)
    ok += good
    print(f"{'ok ' if good else 'BAD'} {question!r} -> {query!r}")
print(f"{sys.argv[1]}: {ok}/{len(cases)}")
