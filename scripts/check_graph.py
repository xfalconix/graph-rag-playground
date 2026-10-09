"""Create one sample snapshot and verify real retrieval and isolation."""

import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
from graph_store import Neo4jStore
from models import QueryPlan
from observability import configure_tracking, save_graph_with_tracking
from sample_data import SAMPLE_GRAPH


def main():
    load_dotenv()
    configure_tracking()
    store = Neo4jStore(os.environ["NEO4J_URI"], os.environ["NEO4J_USERNAME"],
                      os.environ["NEO4J_PASSWORD"], os.getenv("NEO4J_DATABASE", "neo4j"))
    try:
        graph_id = save_graph_with_tracking(store, SAMPLE_GRAPH, uuid.uuid4().hex)
        facts = store.retrieve(graph_id, QueryPlan(intent="shared_actors", first_name="Inception", second_name="The Dark Knight Rises"))
        assert len(facts) == 2 and all(fact["source"] == "Tom Hardy" for fact in facts)
        assert store.retrieve(uuid.uuid4().hex, QueryPlan(intent="movie_cast", first_name="Inception", second_name="")) == []
        Path("artifacts").mkdir(exist_ok=True)
        Path("artifacts/live_graph_id.txt").write_text(graph_id)
        print("Neo4j write, shared-actor retrieval and graph isolation verified.")
    finally:
        store.close()


if __name__ == "__main__":
    main()
