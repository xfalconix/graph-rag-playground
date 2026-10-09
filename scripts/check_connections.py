"""Check configured services without printing secrets or creating graph data."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from graph_store import Neo4jStore
from observability import configure_tracking

load_dotenv()


def check(name, operation):
    try:
        operation()
        print(f"{name}: OK")
    except Exception as error:
        print(f"{name}: FAILED ({type(error).__name__})")


def check_neo4j():
    store = Neo4jStore(os.environ["NEO4J_URI"], os.environ["NEO4J_USERNAME"],
                      os.environ["NEO4J_PASSWORD"], os.getenv("NEO4J_DATABASE", "neo4j"))
    try:
        store.verify()
    finally:
        store.close()


def check_groq():
    llm = ChatGroq(api_key=os.environ["GROQ_API_KEY"], model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
                   max_tokens=50, timeout=20, max_retries=0)
    llm.invoke("Reply with OK.")


if __name__ == "__main__":
    check("Neo4j", check_neo4j)
    check("MLflow", configure_tracking)
    check("Groq", check_groq)
