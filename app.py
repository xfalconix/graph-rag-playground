"""Streamlit interface: create a small graph, ask questions, inspect operations."""

import hmac
import os
import uuid

import graphviz
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError
from groq import RateLimitError

from usage_limits import GRAPH_TTL_HOURS, SESSION_QUESTIONS, SESSION_SAVES, UsageBudget, UsageLimitError

from graph_store import Neo4jStore
from models import MovieGraph
from observability import ask_with_tracking, configure_tracking, read_operations, save_graph_with_tracking
from sample_data import SAMPLE_GRAPH, SAMPLE_QUESTIONS
from workflow import GroqResponder, build_workflow
from version import APP_VERSION


# 1. Read configuration on the server. Secrets never enter the browser or Git.
load_dotenv()
try:
    for key, value in st.secrets.items():
        if isinstance(value, (str, int, float, bool)):
            os.environ[str(key)] = str(value)
except FileNotFoundError:
    pass

st.set_page_config(page_title="GraphRAG Playground", page_icon="🕸️", layout="wide")
st.title("GraphRAG Playground")
st.caption(f"Version {APP_VERSION} · Created by Carlos Falconi · An educational GraphRAG and LLMOps app.")
with st.expander("About this learning app"):
    st.write("Build a small movie graph and see how an answer follows its relationships and evidence.")
    st.markdown("**Streamlit** hosts the interface and Python app. **Neo4j Aura** stores the graph. "
                "**LangGraph** orchestrates planning, retrieval, answering and abstention. "
                "**LangChain** provides the Groq integration and structured outputs. "
                "**MLflow**, hosted in **Databricks Experiments**, records runs, traces, latency, errors and token usage.")
    st.write("**LLM:** " + os.getenv("GROQ_MODEL", "openai/gpt-oss-120b") + " served by Groq.")
    st.caption("A bounded teaching demo, not a general-purpose agent or a complete movie database.")

# A shared demo password limits access while the project is being reviewed.
if os.getenv("APP_PASSWORD"):
    password = st.text_input("Demo password", type="password")
    if not hmac.compare_digest(password, os.environ["APP_PASSWORD"]):
        st.info("Enter the demo password to continue.")
        st.stop()

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex


@st.cache_resource
def get_store():
    return Neo4jStore(os.environ["NEO4J_URI"], os.environ["NEO4J_USERNAME"],
                      os.environ["NEO4J_PASSWORD"], os.getenv("NEO4J_DATABASE", "neo4j"))


@st.cache_resource
def get_budget():
    return UsageBudget(get_store())


@st.cache_resource
def get_experiment():
    return configure_tracking()


def draw_graph(graph: MovieGraph):
    """Render the exact validated snapshot that was saved to Neo4j."""
    dot = graphviz.Digraph()
    dot.attr(rankdir="LR", bgcolor="transparent", pad="0.2")
    colors = {"Person": "#DCECF8", "Movie": "#DDF1EB", "Genre": "#FFF0CF"}
    for entity in graph.entities:
        dot.node(entity.name, label=entity.name, shape="box", style="rounded,filled",
                 fillcolor=colors[entity.kind], color="#B0BCC5", fontname="Arial")
    for relation in graph.relations:
        dot.edge(relation.source, relation.target, label=relation.relation, fontname="Arial", fontsize="10")
    st.graphviz_chart(dot, width="stretch")


required = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "GROQ_API_KEY"]
missing = [name for name in required if not os.getenv(name)]
if missing:
    st.info("Setup is incomplete. Add these server settings: " + ", ".join(missing))

with st.sidebar:
    st.header("Your workspace")
    st.write("**Limits:** 10 entities · 15 relations · 400 characters per question")
    st.write(f"**Questions:** {SESSION_QUESTIONS} attempts per browser session.")
    st.caption("Shared Groq limits: 30 LLM calls/minute · 1,000/day · 8,000 tokens/minute · 200,000/day. "
               "One question uses up to two calls. No monthly cap.")
    st.caption(f"Up to {SESSION_SAVES} graph saves per session. Failed attempts count. "
               "Reopening the app can start a new session; shared limits still apply.")
    st.caption(f"Demo graphs expire after {GRAPH_TTL_HOURS} hours and are cleaned up when the app is used.")
    st.caption("Use only public movie facts or fictional examples. Graphs are stored in Neo4j; "
               "questions and graph facts are sent to Groq. Inputs, answers and operational traces are "
               "recorded in Databricks and visible to the app owner. Graph cleanup does not delete these logs.")
    st.write("**Supported:** directors, cast, shared actors and genres.")
    st.caption("The graph is a small snapshot, not a complete movie database.")

create_tab, ask_tab, operations_tab = st.tabs(["1 · Create graph", "2 · Ask", "3 · Operations"])

# 2. Edit two small tables. Validation happens before any database write.
with create_tab:
    st.subheader("Entities and relationships")
    st.caption("Technology: Streamlit + pandas provide the editable tables; Pydantic validates entity types and links. "
               "The Neo4j Python driver saves a separate graph snapshot, and Graphviz draws its nodes and relationships.")
    with st.form("graph_form"):
        entities = st.data_editor(pd.DataFrame([entity.model_dump() for entity in SAMPLE_GRAPH.entities]),
            num_rows="dynamic", hide_index=True, key="entity_editor", width="stretch",
            column_config={"kind": st.column_config.SelectboxColumn("Type", options=["Person", "Movie", "Genre"], required=True)})
        relations = st.data_editor(pd.DataFrame([relation.model_dump() for relation in SAMPLE_GRAPH.relations]),
            num_rows="dynamic", hide_index=True, key="relation_editor", width="stretch",
            column_config={"relation": st.column_config.SelectboxColumn("Relationship", options=["DIRECTED", "ACTED_IN", "IN_GENRE"], required=True)})
        submitted = st.form_submit_button("Save graph", type="primary", disabled=bool(missing))
    if submitted:
        try:
            graph = MovieGraph.model_validate({"entities": entities.to_dict("records"), "relations": relations.to_dict("records")})
            get_budget().claim("save", st.session_state.session_id)
            get_store().cleanup_expired()
            get_experiment()
            with st.spinner("Saving your graph…"):
                graph_id = save_graph_with_tracking(get_store(), graph, st.session_state.session_id)
            st.session_state.graph_id = graph_id
            st.session_state.saved_graph = graph
            st.session_state.pop("last_result", None)
            st.success("Graph saved. Open the Ask tab to explore it.")
        except UsageLimitError as error:
            st.warning(str(error))
        except ValidationError as error:
            st.error("Please check your tables: " + "; ".join(item["msg"] for item in error.errors()))
        except Exception as error:
            st.error(f"The graph could not be saved ({type(error).__name__}). Check the service connection and MLflow configuration.")
    if "saved_graph" in st.session_state:
        draw_graph(st.session_state.saved_graph)
    else:
        st.caption("Sample graph preview — save it before asking questions.")
        draw_graph(SAMPLE_GRAPH)

# 3. Run the real LangGraph workflow and show its retrieved evidence.
with ask_tab:
    st.subheader("Ask your graph")
    st.caption("Technology: LangGraph runs plan → retrieve → answer (or abstain). LangChain's Groq integration "
               "calls gpt-oss-120b with structured outputs. Fixed Cypher queries retrieve Neo4j facts; "
               "the LLM answers from those facts. MLflow traces each step.")
    st.caption("Example: " + SAMPLE_QUESTIONS[2])
    with st.form("question_form"):
        question = st.text_input("Question", value=SAMPLE_QUESTIONS[0], max_chars=400)
        ask = st.form_submit_button("Ask", type="primary", disabled=bool(missing) or "graph_id" not in st.session_state)
    if "graph_id" not in st.session_state:
        st.info("Save a graph in the first tab to enable questions.")
    if ask:
        st.session_state.pop("last_result", None)
        try:
            if not question.strip():
                raise UsageLimitError("Please enter a question.")
            get_store().cleanup_expired()
            if not get_store().graph_exists(st.session_state.graph_id):
                raise UsageLimitError("This graph has expired. Please save a new graph in the first tab.")
            get_budget().claim("question", st.session_state.session_id)
            get_experiment()
            responder = GroqResponder(os.environ["GROQ_API_KEY"], os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), get_budget())
            workflow = build_workflow(get_store(), responder)
            with st.spinner("Following the graph connections…"):
                st.session_state.last_result = ask_with_tracking(workflow, question, st.session_state.graph_id,
                    [entity.model_dump() for entity in st.session_state.saved_graph.entities], st.session_state.session_id)
        except UsageLimitError as error:
            st.warning(str(error))
        except RateLimitError:
            st.warning("Groq is temporarily at its shared rate limit. Please wait a minute and try again.")
        except Exception as error:
            st.error(f"The question could not be completed ({type(error).__name__}). The operation was attempted; check MLflow for details.")
    if "last_result" in st.session_state:
        result = st.session_state.last_result
        st.write(result["answer"])
        if result["facts"]:
            st.dataframe(pd.DataFrame(result["facts"]), hide_index=True, width="stretch")
        st.caption("Result: " + result["status"] + " · Trace: " + result["trace_id"])

# 4. Read persisted MLflow runs instead of inventing dashboard data.
with operations_tab:
    st.subheader("Your session's operations")
    st.caption("Technology: MLflow stores runs, metrics and nested traces in Databricks Experiments. "
               "The app reads them back with the MLflow API; pandas and Streamlit display this session's table and chart. "
               "Databricks is the tracking service here; it does not host the LLM or the app.")
    st.caption("Real demo executions. This panel reports performance, not answer accuracy.")
    if st.button("Refresh operations", disabled=bool(missing)):
        try:
            st.session_state.operations = read_operations(get_experiment(), st.session_state.session_id)
        except Exception as error:
            st.error(f"Metrics could not be loaded ({type(error).__name__}).")
    operations = st.session_state.get("operations", pd.DataFrame())
    if operations.empty:
        st.info("Save a graph or ask a question, then refresh this panel.")
    else:
        queries = operations[operations.get("tags.operation", "") == "ask_graph"]
        first, second, third = st.columns(3)
        first.metric("Questions", len(queries))
        duration = queries.get("metrics.duration_seconds", pd.Series(dtype=float)).dropna()
        second.metric("Median response time", f"{duration.median():.2f} s" if len(duration) else "—")
        errors = queries.get("metrics.error", pd.Series(dtype=float)).fillna(0)
        third.metric("Failed questions", int(errors.sum()))
        if len(duration):
            chart = queries[["start_time", "metrics.duration_seconds"]].dropna().rename(columns={"metrics.duration_seconds": "Seconds"})
            # Each point is one real request; a single request remains visible.
            st.scatter_chart(chart.set_index("start_time")["Seconds"])
        wanted = ["start_time", "tags.operation", "tags.result", "metrics.duration_seconds", "metrics.fact_count", "tags.trace_id"]
        st.dataframe(operations[[column for column in wanted if column in operations]], hide_index=True, width="stretch")
        st.caption("Open the MLflow experiment to inspect nested traces and token usage where available.")
