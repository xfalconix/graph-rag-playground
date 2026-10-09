"""Behavior checks: graph limits, safe routing, abstention and evidence."""

import pytest
from pydantic import ValidationError

from models import GroundedAnswer, MovieGraph, QueryPlan
from sample_data import SAMPLE_GRAPH
from workflow import build_workflow


class FakeStore:
    def __init__(self):
        self.queries = []

    def retrieve(self, graph_id, plan):
        self.queries.append((graph_id, plan))
        if graph_id != "my-graph" or plan.intent != "movie_cast":
            return []
        return [{"id": "F1", "source": "Tom Hardy", "relation": "ACTED_IN", "target": "Inception"}]


class FakeResponder:
    def __init__(self, first_name="Inception", evidence_ids=None):
        self.first_name = first_name
        self.evidence_ids = ["F1"] if evidence_ids is None else evidence_ids
        self.answer_calls = 0

    def plan(self, question, entities):
        return QueryPlan(intent="movie_cast", first_name=self.first_name, second_name="")

    def answer(self, question, facts):
        self.answer_calls += 1
        return GroundedAnswer(answer="Tom Hardy acted in Inception.", evidence_ids=self.evidence_ids)


def invoke(store, responder, graph_id="my-graph", question="Who acted in Inception?"):
    return build_workflow(store, responder).invoke({
        "question": question, "graph_id": graph_id,
        "entities": [entity.model_dump() for entity in SAMPLE_GRAPH.entities],
    })


def test_graph_rejects_more_than_ten_entities():
    data = SAMPLE_GRAPH.model_dump()
    data["entities"].extend({"name": f"Extra {index}", "kind": "Person"} for index in range(3))
    with pytest.raises(ValidationError):
        MovieGraph.model_validate(data)


@pytest.mark.parametrize("change", ["missing_endpoint", "wrong_type", "duplicate_edge", "duplicate_name"])
def test_invalid_graph_is_rejected_before_storage(change):
    data = SAMPLE_GRAPH.model_dump()
    if change == "missing_endpoint":
        data["relations"][0]["target"] = "Missing movie"
    elif change == "wrong_type":
        data["relations"][0]["source"] = "Inception"
    elif change == "duplicate_edge":
        data["relations"].append(data["relations"][0])
    else:
        data["entities"].append({"name": "inception", "kind": "Movie"})
    with pytest.raises(ValidationError):
        MovieGraph.model_validate(data)


def test_question_passes_the_server_graph_id_to_storage():
    store = FakeStore()
    result = invoke(store, FakeResponder())
    assert result["status"] == "answered"
    assert store.queries[0][0] == "my-graph"
    assert result["evidence_ids"] == ["F1"]


def test_missing_evidence_abstains_without_answer_generation():
    responder = FakeResponder()
    result = invoke(FakeStore(), responder, graph_id="other-graph")
    assert result["status"] == "abstained"
    assert responder.answer_calls == 0


def test_invented_entity_does_not_become_an_executable_query():
    store = FakeStore()
    result = invoke(store, FakeResponder(first_name="' MATCH (n) DETACH DELETE n //"))
    assert result["status"] == "abstained"
    assert store.queries[0][1].intent == "unsupported"


def test_answer_with_fabricated_evidence_id_is_rejected():
    result = invoke(FakeStore(), FakeResponder(evidence_ids=["F999"]))
    assert result["status"] == "invalid_evidence"


def test_question_length_is_enforced_in_workflow():
    with pytest.raises(ValueError, match="400"):
        invoke(FakeStore(), FakeResponder(), question="x" * 401)
