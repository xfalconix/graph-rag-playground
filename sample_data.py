"""A deliberately small set of movie facts for the first walkthrough."""

from models import MovieGraph


SAMPLE_GRAPH = MovieGraph.model_validate({
    "entities": [
        {"name": "Christopher Nolan", "kind": "Person"},
        {"name": "Leonardo DiCaprio", "kind": "Person"},
        {"name": "Tom Hardy", "kind": "Person"},
        {"name": "Christian Bale", "kind": "Person"},
        {"name": "Inception", "kind": "Movie"},
        {"name": "The Dark Knight Rises", "kind": "Movie"},
        {"name": "Science fiction", "kind": "Genre"},
        {"name": "Action", "kind": "Genre"},
    ],
    "relations": [
        {"source": "Christopher Nolan", "relation": "DIRECTED", "target": "Inception"},
        {"source": "Christopher Nolan", "relation": "DIRECTED", "target": "The Dark Knight Rises"},
        {"source": "Leonardo DiCaprio", "relation": "ACTED_IN", "target": "Inception"},
        {"source": "Tom Hardy", "relation": "ACTED_IN", "target": "Inception"},
        {"source": "Tom Hardy", "relation": "ACTED_IN", "target": "The Dark Knight Rises"},
        {"source": "Christian Bale", "relation": "ACTED_IN", "target": "The Dark Knight Rises"},
        {"source": "Inception", "relation": "IN_GENRE", "target": "Science fiction"},
        {"source": "The Dark Knight Rises", "relation": "IN_GENRE", "target": "Action"},
    ],
})

SAMPLE_QUESTIONS = [
    "Which movies did Christopher Nolan direct?",
    "Who acted in Inception?",
    "Which actors are shared by Inception and The Dark Knight Rises?",
    "What genre is Inception?",
    "Who acted in Titanic?",
]
