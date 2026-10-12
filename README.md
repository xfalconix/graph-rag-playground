# GraphRAG Playground · v1.0

Created by **Carlos Falconi**. A small learning application for building a movie graph, asking grounded questions, and inspecting real execution traces.

**Streamlit → LangGraph → Neo4j + Groq → MLflow**

[Open the public demo](https://graph-rag-playground.streamlit.app/)

## What the demo does

1. Edit up to 10 entities and 15 relationships.
2. Save an isolated graph snapshot in Neo4j and see its diagram.
3. Ask about directors, casts, shared actors or genres.
4. Inspect the retrieved facts, answer, execution time and MLflow trace.

The LLM selects a supported query intent and entity names. The application validates them and executes a fixed, parameterized Cypher query. It never executes Cypher written by the LLM. Missing evidence leads to an explicit abstention.

## Start locally

Use Python 3.11. Create a virtual environment, install the dependencies, and copy `.env.example` to `.env`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

Configure Neo4j, Groq and MLflow in `.env`. For local MLflow, set `MLFLOW_TRACKING_URI=sqlite:///mlflow.db` and `MLFLOW_EXPERIMENT_NAME=graph-rag-playground`. For Databricks, use the workspace host without a browser query string and authenticated access to an experiment.

```bash
python scripts/check_connections.py
python scripts/check_pipeline.py
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

## Read the walkthrough

Open `notebooks/01_graph_rag_walkthrough.ipynb`. It follows the same small functions as the app: data → validation → graph storage → retrieval → LangGraph → answer → MLflow.

This is a learning prototype. The walkthrough explains the implementation; the checks document its behavior without claiming production reliability.

## Files

| File | Responsibility |
|---|---|
| `models.py` | Input limits and typed relationships |
| `sample_data.py` | Small, editable sample graph |
| `graph_store.py` | Session-scoped Neo4j writes and fixed reads |
| `workflow.py` | LangGraph routing and grounded generation |
| `observability.py` | MLflow runs, traces and dashboard queries |
| `usage_limits.py` | Persistent, atomic Groq budgets and per-session limits |
| `app.py` | Three-tab interface |

## Deployment

Streamlit Community Cloud can run `app.py` from a private GitHub repository. Configure secrets in the hosting settings; never commit `.env` or `.streamlit/secrets.toml`. Set `APP_PASSWORD` while the demo is under review. Deploying the app does not require making the repository public.

Databricks Apps is an alternative for authenticated demonstrations, subject to Free Edition networking and runtime limits. Neo4j and the LLM remain separate services.

## Limits and evidence

- A small user-provided snapshot is not a complete movie database.
- Evidence IDs make an answer inspectable; they do not prove factual correctness. Evaluate answers against known cases.
- Only four question types are supported. This is a bounded workflow, not a general agent.
- The operations tab reads persisted MLflow runs for the current session. A new browser session has a new ID; earlier runs remain in MLflow.
- Graph snapshots use the `GRPlayEntity` label and opaque graph IDs. Existing database data is not modified. Snapshots expire after 24 hours. Cleanup runs on save/question activity, not on a schedule; a sleeping app cleans up when next used. Legacy snapshots get 24 hours of grace. Only isolated `GRPlayEntity` nodes are deleted. MLflow history is retained separately; this is not a production tenancy guarantee.
- The history represents demo/test traffic, not sustained production usage. MLflow tracks operations; it does not measure uptime while the app is asleep.
- API keys and limits belong to the configured service accounts. The frontend never receives those credentials.

## Public-demo usage controls

- 10 question attempts and 3 graph saves per browser session. Reopening the browser can create a new session; this is not an authenticated per-person quota.
- Shared Groq limits for `openai/gpt-oss-120b`: 30 calls/minute, 1,000/day, 8,000 tokens/minute, 200,000/day. There is **no monthly cap**. One question makes one or two model calls.
- A unique Neo4j control node and write transaction serialize reservations across users and restarts. Rolling 60-second and 24-hour windows avoid midnight bursts. Failed attempts count.
- Each LLM call first reserves an intentionally conservative UTF-8 input estimate plus maximum output and a formatting margin. Provider-reported tokens replace it after a response. Ambiguous failures retain their reservation. Automatic LLM retries are disabled.
- Groq is authoritative: other apps share the organization's quota, token estimates differ from exact accounting, and a 429 can still occur. The app explains the temporary refusal. No LLM call is made if the budget store is unavailable.
- Graph saves have a separate shared cap of 100 per rolling day to bound storage growth. The control ledger is separate from expiring graph snapshots.
- Questions, supplied facts and answers are recorded in the owner's Databricks experiment. Only public/fictitious inputs are appropriate. Removing a graph does not erase existing traces.

Limits were verified in the account on 2026-10-09 and in [Groq's official limits](https://console.groq.com/docs/rate-limits). Databricks supplies managed MLflow tracking here; it does not run this Streamlit application, host Groq, or train a model. Experiment-backed tracing currently has a [100,000-trace limit](https://docs.databricks.com/aws/en/resources/limits), distinct from the 500,000-run limit.

## Validation

```bash
pip install -r requirements-dev.txt
pytest -q
```

Tests cover input limits, invalid relationships, unknown entities, evidence validation, abstention, run persistence, interface startup, session limits, rolling windows and refusal before an LLM call. Live service checks and an end-to-end question must also pass before deployment is considered verified.

`python scripts/check_limits.py` checks concurrent reservations and scoped expiry against Neo4j using temporary verification records. It preserves unrelated data.

`check_pipeline.py` makes real Groq calls and writes a small Neo4j snapshot. It checks a shared actor question, an abstention, and read-back of MLflow runs/traces. It labels this traffic `live_check`; these two smoke checks are not an accuracy benchmark. Runs also record the prompt version, model, app version, and Git revision (or the optional `GIT_COMMIT` setting).

## Inspiration and sources

The clear notebook progression was inspired by [DSwithBappy's GraphRAG tutorial](https://github.com/entbappy/Complete-Agentic-AI-Course/tree/main/Advanced-RAG/Graph-RAG). This project is an independently written implementation; the tutorial has not been copied or bundled.

- [Neo4j Python driver](https://neo4j.com/docs/python-manual/current/)
- [LangGraph workflows](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
- [MLflow tracing](https://mlflow.org/docs/latest/genai/tracing/)
- [Streamlit deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud)

Sample statements are short movie facts entered for demonstration, not copied plot summaries or film scripts.
