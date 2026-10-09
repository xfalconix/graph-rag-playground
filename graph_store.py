"""Neo4j persistence with a separate namespace and bounded read queries."""

import hashlib
import json
import uuid

import mlflow
from neo4j import GraphDatabase, Query, unit_of_work

from models import MovieGraph, QueryPlan


# The model cannot modify these queries or introduce a write statement.
READ_QUERIES = {
    "director_movies": """
        MATCH (a:GRPlayEntity {graph_id: $graph_id, name: $first_name, kind: 'Person'})
              -[r:DIRECTED]->(b:GRPlayEntity {graph_id: $graph_id, kind: 'Movie'})
        RETURN a.name AS source, type(r) AS relation, b.name AS target
        ORDER BY target LIMIT 15
    """,
    "movie_cast": """
        MATCH (a:GRPlayEntity {graph_id: $graph_id, kind: 'Person'})
              -[r:ACTED_IN]->(b:GRPlayEntity {graph_id: $graph_id, name: $first_name, kind: 'Movie'})
        RETURN a.name AS source, type(r) AS relation, b.name AS target
        ORDER BY source LIMIT 15
    """,
    "movie_genres": """
        MATCH (a:GRPlayEntity {graph_id: $graph_id, name: $first_name, kind: 'Movie'})
              -[r:IN_GENRE]->(b:GRPlayEntity {graph_id: $graph_id, kind: 'Genre'})
        RETURN a.name AS source, type(r) AS relation, b.name AS target
        ORDER BY target LIMIT 15
    """,
    "shared_actors": """
        MATCH (first:GRPlayEntity {graph_id: $graph_id, name: $first_name, kind: 'Movie'})
              <-[:ACTED_IN]-(actor:GRPlayEntity {graph_id: $graph_id, kind: 'Person'})
              -[:ACTED_IN]->(second:GRPlayEntity {graph_id: $graph_id, name: $second_name, kind: 'Movie'})
        UNWIND [first, second] AS movie
        RETURN actor.name AS source, 'ACTED_IN' AS relation, movie.name AS target
        ORDER BY source, target LIMIT 30
    """,
}


def graph_fingerprint(graph: MovieGraph) -> str:
    """Identify the input data without using the session identifier."""
    canonical = json.dumps(graph.model_dump(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


class Neo4jStore:
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(username, password), connection_timeout=10)
        self.database = database

    def verify(self):
        self.driver.verify_connectivity()

    @mlflow.trace(name="save_graph", span_type="TOOL")
    def save(self, graph: MovieGraph) -> str:
        # A new opaque ID creates an immutable snapshot. Existing data is untouched.
        graph_id = uuid.uuid4().hex
        with self.driver.session(database=self.database) as session:
            session.execute_write(self._write_snapshot, graph_id, graph)
        return graph_id

    @staticmethod
    @unit_of_work(timeout=15)
    def _write_snapshot(transaction, graph_id: str, graph: MovieGraph):
        transaction.run("""
            UNWIND $entities AS entity
            CREATE (:GRPlayEntity {graph_id: $graph_id, name: entity.name, kind: entity.kind})
        """, graph_id=graph_id, entities=[entity.model_dump() for entity in graph.entities]).consume()
        for relation_type in ("DIRECTED", "ACTED_IN", "IN_GENRE"):
            relations = [relation.model_dump() for relation in graph.relations if relation.relation == relation_type]
            # Only the fixed allowlist above can enter this query string.
            transaction.run(f"""
                UNWIND $relations AS relation
                MATCH (source:GRPlayEntity {{graph_id: $graph_id, name: relation.source}})
                MATCH (target:GRPlayEntity {{graph_id: $graph_id, name: relation.target}})
                CREATE (source)-[:{relation_type} {{graph_id: $graph_id}}]->(target)
            """, graph_id=graph_id, relations=relations).consume()

    @mlflow.trace(name="retrieve_graph_facts", span_type="RETRIEVER")
    def retrieve(self, graph_id: str, plan: QueryPlan) -> list[dict]:
        if plan.intent not in READ_QUERIES:
            return []
        with self.driver.session(database=self.database, default_access_mode="READ") as session:
            records = session.run(
                Query(READ_QUERIES[plan.intent], timeout=10),
                graph_id=graph_id,
                first_name=plan.first_name,
                second_name=plan.second_name,
            ).data()
        return [{"id": f"F{index + 1}", **record} for index, record in enumerate(records)]

    def close(self):
        self.driver.close()
