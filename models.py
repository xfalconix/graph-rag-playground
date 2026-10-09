"""Small, explicit schemas for the graph and the supported questions."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Entity(BaseModel):
    """A named entity in one visitor's graph."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    name: str = Field(min_length=1, max_length=80)
    kind: Literal["Person", "Movie", "Genre"]


class Relation(BaseModel):
    """A directed connection between two named entities."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    source: str = Field(min_length=1, max_length=80)
    relation: Literal["DIRECTED", "ACTED_IN", "IN_GENRE"]
    target: str = Field(min_length=1, max_length=80)


class MovieGraph(BaseModel):
    """Validate limits and relationships before writing anything to Neo4j."""

    entities: list[Entity] = Field(min_length=1, max_length=10)
    relations: list[Relation] = Field(max_length=15)

    @model_validator(mode="after")
    def validate_connections(self):
        entity_types = {entity.name: entity.kind for entity in self.entities}
        names = [entity.name.casefold() for entity in self.entities]
        if len(names) != len(set(names)):
            raise ValueError("Entity names must be unique, ignoring letter case.")
        allowed_pairs = {
            "DIRECTED": ("Person", "Movie"),
            "ACTED_IN": ("Person", "Movie"),
            "IN_GENRE": ("Movie", "Genre"),
        }
        seen = set()
        for relation in self.relations:
            pair = (entity_types.get(relation.source), entity_types.get(relation.target))
            if pair != allowed_pairs[relation.relation]:
                raise ValueError(f"Invalid endpoints for {relation.relation}: {relation.source} → {relation.target}.")
            key = (relation.source, relation.relation, relation.target)
            if key in seen:
                raise ValueError("Duplicate relationships are not allowed.")
            seen.add(key)
        return self


class QueryPlan(BaseModel):
    """The LLM selects an intent and names, never executable Cypher."""

    model_config = ConfigDict(extra="forbid")
    intent: Literal["director_movies", "movie_cast", "shared_actors", "movie_genres", "unsupported"]
    first_name: str
    second_name: str


class GroundedAnswer(BaseModel):
    """Require the answer to identify the retrieved facts it uses."""

    model_config = ConfigDict(extra="forbid")
    answer: str
    evidence_ids: list[str]
