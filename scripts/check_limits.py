"""Verify atomic reservations and scoped cleanup against the configured Neo4j."""

import json
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv

from graph_store import Neo4jStore
from sample_data import SAMPLE_GRAPH
from usage_limits import UsageBudget, UsageLimitError


def main():
    load_dotenv()
    store = Neo4jStore(os.environ["NEO4J_URI"], os.environ["NEO4J_USERNAME"],
                       os.environ["NEO4J_PASSWORD"], os.getenv("NEO4J_DATABASE", "neo4j"))
    key = "verification-" + uuid.uuid4().hex
    budget = UsageBudget(store, key)
    try:
        def attempt(index):
            try:
                budget.claim("llm", tokens=1)
                return True
            except UsageLimitError:
                return False

        with ThreadPoolExecutor(max_workers=8) as pool:
            accepted = sum(pool.map(attempt, range(40)))
        assert accepted == 30, f"Concurrency escaped the RPM cap: {accepted}"
        second_client = UsageBudget(store, key)
        try:
            second_client.claim("llm", tokens=1)
        except UsageLimitError:
            pass
        else:
            raise AssertionError("A new client reset the persistent limit")

        # Only this fixture is deliberately expired. Other demo graphs get a grace period.
        graph_id = store.save(SAMPLE_GRAPH)
        with store.driver.session(database=store.database) as session:
            before = session.run("MATCH(n) WHERE NOT n:GRPlayEntity AND NOT n:GRPlayControl "
                                 "RETURN count(n) AS n").single()["n"]
            session.run("MATCH(n:GRPlayEntity {graph_id:$id}) "
                        "SET n.expires_at=datetime()-duration({hours:1})", id=graph_id).consume()
        assert not store.graph_exists(graph_id)
        removed = store.cleanup_expired()
        assert removed >= 8
        with store.driver.session(database=store.database) as session:
            assert session.run("MATCH(n:GRPlayEntity {graph_id:$id}) RETURN count(n) AS n",
                               id=graph_id).single()["n"] == 0
            after = session.run("MATCH(n) WHERE NOT n:GRPlayEntity AND NOT n:GRPlayControl "
                                "RETURN count(n) AS n").single()["n"]
            assert after == before
        print("PASS: 40 concurrent reservations accepted exactly 30; a new client kept the limit.")
        print("PASS: expired fixture removed; unrelated node count preserved:", after)
    finally:
        with store.driver.session(database=store.database) as session:
            session.run("MATCH(n:GRPlayControl {key:$key}) DELETE n", key=key).consume()
        store.close()


if __name__ == "__main__":
    main()
