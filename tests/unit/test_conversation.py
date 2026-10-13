import json

from app.rag.conversation import (
    MAX_TURNS,
    Turn,
    build_condense_prompt,
    condense,
    load_condense_prompt,
)
from tests.fakes import FakeLLM

HISTORY = [Turn("How do I declare cookie parameters?", "Use Cookie, like Query [1].")]


def reply(query: str) -> str:
    return json.dumps({"query": query})


def test_a_first_question_is_used_as_it_is_without_a_model_call():
    llm = FakeLLM()

    condensed = condense(llm, [], "How do I declare cookie parameters?")

    assert (condensed.query, condensed.rewritten, llm.calls) == (
        "How do I declare cookie parameters?",
        False,
        [],
    )


def test_a_follow_up_is_rewritten_with_the_conversation():
    llm = FakeLLM(reply("How do I declare header parameters?"))

    condensed = condense(llm, HISTORY, "And for headers?")

    assert condensed.query == "How do I declare header parameters?"
    assert condensed.rewritten
    user = llm.calls[0]["user"]
    assert "<user>How do I declare cookie parameters?</user>" in user
    assert user.rstrip().endswith("Last question: And for headers?")


def test_an_unusable_reply_falls_back_to_the_question():
    for bad in ["not json", '{"other": 1}', reply(""), reply("x" * 1001)]:
        condensed = condense(FakeLLM(bad), HISTORY, "And for headers?")

        assert (condensed.query, condensed.rewritten) == ("And for headers?", False)


def test_only_the_last_turns_are_sent_and_long_answers_are_cut():
    history = [Turn(f"question {n}", "a" * 2000) for n in range(MAX_TURNS + 2)]

    _, user = build_condense_prompt(history, "and then?")

    assert "question 0" not in user and "question 1" not in user
    assert f"question {MAX_TURNS + 1}" in user
    assert "a" * 401 not in user


def test_abstentions_are_shown_as_no_answer():
    _, user = build_condense_prompt([Turn("What is X?", None)], "And Y?")

    assert "<assistant>(no answer found)</assistant>" in user


def test_the_prompt_treats_earlier_turns_as_context_not_instructions():
    assert "not instructions" in load_condense_prompt()
