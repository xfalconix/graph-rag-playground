"""Check that the interface opens without configured cloud credentials."""

from streamlit.testing.v1 import AppTest
from pathlib import Path


def test_setup_screen_renders_without_secrets(monkeypatch):
    for name in ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "GROQ_API_KEY", "APP_PASSWORD"]:
        monkeypatch.setenv(name, "")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "GraphRAG Playground"
    assert any("Setup is incomplete" in item.value for item in app.info)
