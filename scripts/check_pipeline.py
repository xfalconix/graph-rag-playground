"""Run two real questions and verify their persisted MLflow runs and traces."""

import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
from dotenv import load_dotenv

from graph_store import Neo4jStore
from observability import ask_with_tracking, configure_tracking, read_operations, save_graph_with_tracking
from sample_data import SAMPLE_GRAPH
from workflow import GroqResponder, build_workflow
from usage_limits import UsageBudget


def main():
    load_dotenv()
    os.environ["TRAFFIC_TYPE"] = "live_check"
    experiment_id = configure_tracking()
    session_id = uuid.uuid4().hex
    store = Neo4jStore(os.environ["NEO4J_URI"], os.environ["NEO4J_USERNAME"],
                      os.environ["NEO4J_PASSWORD"], os.getenv("NEO4J_DATABASE", "neo4j"))
    try:
        budget = UsageBudget(store)
        budget.claim("save", session_id)
        graph_id = save_graph_with_tracking(store, SAMPLE_GRAPH, session_id)
        responder = GroqResponder(os.environ["GROQ_API_KEY"], os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), budget)
        workflow = build_workflow(store, responder)
        entities = [entity.model_dump() for entity in SAMPLE_GRAPH.entities]

        # A two-hop question must retrieve both movie relationships for Tom Hardy.
        budget.claim("question", session_id)
        answer = ask_with_tracking(workflow,
            "Who acted in both Inception and The Dark Knight Rises?",
            graph_id, entities, session_id)
        assert answer["status"] == "answered", "The answer was not grounded."
        assert len(answer["facts"]) == 2, "Expected two retrieved relationships."
        assert {fact["source"] for fact in answer["facts"]} == {"Tom Hardy"}
        assert "Tom Hardy" in answer["answer"], "The answer omitted the known shared actor."

        # An absent entity must not be answered using the LLM's general knowledge.
        budget.claim("question", session_id)
        absent = ask_with_tracking(workflow, "Who acted in Titanic?", graph_id, entities, session_id)
        assert absent["status"] == "abstained", "Expected abstention for an absent movie."

        runs = read_operations(experiment_id, session_id)
        assert len(runs) == 3, "Expected one graph run and two question runs."
        for result in (answer, absent):
            trace = mlflow.get_trace(result["trace_id"], flush=True)
            assert trace is not None, "The trace could not be read back."
        print("Real graph write, grounded answer, abstention, runs and traces verified.")
        print(f"Experiment ID: {experiment_id}; session ID: {session_id}")
        print("These are two smoke checks, not a measured accuracy benchmark.")
    finally:
        store.close()


if __name__ == "__main__":
    main()
