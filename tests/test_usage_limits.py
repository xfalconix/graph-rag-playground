"""Check shared budgets, time boundaries, and the real LLM call boundary."""

from types import SimpleNamespace

import pytest

from models import QueryPlan
from usage_limits import GROQ_LIMITS, UsageLimitError, reserve
from workflow import GroqResponder


SESSION = "a" * 32
NOW = 1_800_000_000.0


def test_session_limit_does_not_reset_on_new_day():
    state = {}
    for i in range(10):
        state = reserve(state, "question", SESSION, NOW + i)
    with pytest.raises(UsageLimitError, match="10"):
        reserve(state, "question", SESSION, NOW + 86401)
    assert state["sessions"][SESSION]["question"] == 10
    assert reserve(state, "question", "b" * 32, NOW + 86401)


@pytest.mark.parametrize("scope,count,age", [("rpm", 30, 10), ("rpd", 1000, 120)])
def test_global_request_limit_across_sessions(scope, count, age):
    state = {"calls": [{"id": str(i), "time": NOW - age, "tokens": 1} for i in range(count)]}
    with pytest.raises(UsageLimitError):
        reserve(state, "llm", "b" * 32, NOW, 1, "new")


@pytest.mark.parametrize("tokens,age", [(8000, 10), (200000, 120)])
def test_token_budget_refuses_before_sending(tokens, age):
    state = {"calls": [{"id": "old", "time": NOW - age, "tokens": tokens}]}
    with pytest.raises(UsageLimitError):
        reserve(state, "llm", "", NOW, 1, "new")
    assert len(state["calls"]) == 1


def test_rolling_windows_expire_without_month_reset():
    state = {"calls": [{"id": "old", "time": NOW - 86400, "tokens": 200000}]}
    updated = reserve(state, "llm", "", NOW, 8000, "new")
    assert len(updated["calls"]) == 1
    assert "monthly" not in updated
    updated = reserve(updated, "llm", "", NOW + 60, 8000, "next")
    assert len(updated["calls"]) == 2


def test_rejected_reservation_does_not_mutate_ledger():
    state = {"calls": []}
    with pytest.raises(UsageLimitError):
        reserve(state, "llm", "", NOW, GROQ_LIMITS["tpm"] + 1, "new")
    assert state == {"calls": []}


def test_graph_save_limit_is_independent_of_questions():
    state = {}
    for _ in range(3):
        state = reserve(state, "save", SESSION, NOW)
    with pytest.raises(UsageLimitError, match="3"):
        reserve(state, "save", SESSION, NOW)
    assert reserve(state, "question", SESSION, NOW)


def test_refusal_never_invokes_llm():
    class Budget:
        def claim(self, *args, **kwargs):
            raise UsageLimitError("full")

    class Runnable:
        def invoke(self, messages):
            pytest.fail("Provider must not be called after a refusal")

    responder = object.__new__(GroqResponder)
    responder.budget = Budget()
    with pytest.raises(UsageLimitError):
        responder._invoke(Runnable(), [("human", "hi")], QueryPlan)


def test_provider_usage_reconciles_reserved_tokens():
    calls = []

    class Budget:
        def claim(self, *args, **kwargs):
            return "request-id"

        def settle(self, request_id, tokens):
            calls.append((request_id, tokens))

    result = QueryPlan(intent="unsupported", first_name="", second_name="")

    class Runnable:
        def invoke(self, messages):
            return {"raw": SimpleNamespace(usage_metadata={"total_tokens": 123}),
                    "parsed": result, "parsing_error": None}

    responder = object.__new__(GroqResponder)
    responder.budget = Budget()
    assert responder._invoke(Runnable(), [("human", "hi")], QueryPlan) == result
    assert calls == [("request-id", 123)]
