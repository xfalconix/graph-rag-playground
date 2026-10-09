"""Persistent Groq-shaped budgets shared by every app session and restart."""

import copy
import json
import time
import uuid

from neo4j import unit_of_work


SESSION_QUESTIONS = 10
SESSION_SAVES = 3
DAILY_SAVES = 100
GROQ_LIMITS = {"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200_000}
GRAPH_TTL_HOURS = 24
MAX_OUTPUT_TOKENS = 2048


class UsageLimitError(ValueError):
    """An expected budget refusal, safe to display to the visitor."""


def reserve(state: dict, action: str, session_id: str, now: float,
            tokens: int = 0, request_id: str = "") -> dict:
    """Return an updated ledger or refuse without changing the original.

    Calls use rolling 60-second/24-hour windows, including across midnight.
    A request reserves its estimated prompt plus maximum output before Groq.
    Failed/ambiguous calls retain that reservation. No monthly budget exists.
    """
    state = copy.deepcopy(state)
    state["calls"] = [c for c in state.get("calls", []) if c["time"] > now - 86400]
    state["saves"] = [t for t in state.get("saves", []) if t > now - 86400]
    # Keep active session counters; remove only sessions inactive for 30 days.
    state["sessions"] = {k: v for k, v in state.get("sessions", {}).items()
                         if v["last_seen"] > now - 30 * 86400}
    if action == "llm":
        if tokens <= 0 or not request_id:
            raise ValueError("A positive token reservation and request ID are required.")
        if tokens > GROQ_LIMITS["tpm"]:
            raise UsageLimitError("This input is too large for the demo token budget. Use shorter names or a shorter question.")
        minute = [c for c in state["calls"] if c["time"] > now - 60]
        for calls, request_limit, token_limit, wait in [
            (minute, GROQ_LIMITS["rpm"], GROQ_LIMITS["tpm"], "60 seconds"),
            (state["calls"], GROQ_LIMITS["rpd"], GROQ_LIMITS["tpd"], "up to 24 hours"),
        ]:
            if len(calls) >= request_limit or sum(c["tokens"] for c in calls) + tokens > token_limit:
                raise UsageLimitError(f"The shared Groq demo budget is busy. Please wait {wait} and try again.")
        state["calls"].append({"time": now, "tokens": tokens, "id": request_id})
        return state
    if action not in ("question", "save"):
        raise ValueError("Unknown budget action.")
    if len(session_id) != 32 or any(c not in "0123456789abcdef" for c in session_id):
        raise ValueError("Invalid session identifier.")
    session = state["sessions"].setdefault(session_id, {"last_seen": now})
    limit = SESSION_QUESTIONS if action == "question" else SESSION_SAVES
    if session.get(action, 0) >= limit:
        raise UsageLimitError(f"This demo allows {limit} {action} attempts per browser session. The limit has been reached.")
    if action == "save" and len(state["saves"]) >= DAILY_SAVES:
        raise UsageLimitError("The demo's shared graph-save limit is reached. Please return tomorrow.")
    session[action] = session.get(action, 0) + 1
    session["last_seen"] = now
    if action == "save":
        state["saves"].append(now)
    return state


class UsageBudget:
    """One unique Neo4j node serializes reservations with a write lock."""

    def __init__(self, store, key: str = "public-demo-v1"):
        self.store, self.key = store, key
        with store.driver.session(database=store.database) as session:
            session.run("CREATE CONSTRAINT grplay_control_key IF NOT EXISTS "
                        "FOR (n:GRPlayControl) REQUIRE n.key IS UNIQUE").consume()

    def claim(self, action: str, session_id: str = "", tokens: int = 0) -> str:
        request_id = uuid.uuid4().hex
        with self.store.driver.session(database=self.store.database) as session:
            session.execute_write(self._update, action, session_id, tokens, request_id)
        return request_id

    def settle(self, request_id: str, actual_tokens: int):
        """Replace a reservation with provider-reported input + output tokens."""
        if actual_tokens <= 0:
            return
        with self.store.driver.session(database=self.store.database) as session:
            session.execute_write(self._update, "settle", "", actual_tokens, request_id)

    @unit_of_work(timeout=15)
    def _update(self, transaction, action, session_id, tokens, request_id):
        # Acquire the write lock before reading; competing requests must wait.
        record = transaction.run("""
            MERGE (n:GRPlayControl {key: $key})
            SET n.lock_version = coalesce(n.lock_version, 0) + 1
            RETURN n.payload AS payload
        """, key=self.key).single()
        state = json.loads(record["payload"] or "{}")
        if action == "settle":
            for call in state.get("calls", []):
                if call["id"] == request_id:
                    call["tokens"] = tokens
        else:
            state = reserve(state, action, session_id, time.time(), tokens, request_id)
        transaction.run("MATCH (n:GRPlayControl {key: $key}) SET n.payload = $payload",
                        key=self.key, payload=json.dumps(state)).consume()


def token_reservation(messages: list, schema: type) -> int:
    """Estimate prompt/tool tokens + maximum output + formatting margin.

    Tokenization is an estimate; Groq remains authoritative and can return 429,
    especially when other apps in the same organization consume its quota.
    """
    from langchain_core.utils.function_calling import convert_to_openai_tool

    payload = json.dumps({"messages": messages, "tools": [convert_to_openai_tool(schema)]}, ensure_ascii=False)
    # UTF-8 byte count deliberately overestimates byte-pair input tokens.
    # No tokenizer download is needed at app startup.
    prompt_tokens = len(payload.encode("utf-8"))
    return prompt_tokens + MAX_OUTPUT_TOKENS + 256
