# Conversational AI — data generation & NL querying

Python stack for a **Streamlit** app that calls **Gemini 2.0 Flash** on **Vertex AI** (via the **Google Gen AI SDK**), uses **PostgreSQL**, optional **Langfuse** tracing, and is containerized with **Docker**. **SonarQube** is configured via `sonar-project.properties`.

## Prerequisites

- **Python 3.11+**
- **Google Cloud** project with **Vertex AI API** enabled and billing (if required by your org)
- **Gemini on Vertex**: your principal needs a role such as **Vertex AI User** (`roles/aiplatform.user`)
- **Docker** (optional, for Compose)

## Quick start (local)

1. Copy environment template and set your GCP project:

   ```bash
   cp .env.example .env
   # Edit .env: GOOGLE_CLOUD_PROJECT, optional LANGFUSE_*
   ```

2. Authenticate for Vertex (Application Default Credentials):

   ```bash
   gcloud config set project YOUR_PROJECT_ID
   gcloud auth application-default login
   ```

3. Create a venv and install dependencies:

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

4. Start PostgreSQL (e.g. via Docker):

   ```bash
   docker compose up -d db
   ```

5. Run the UI:

   ```bash
   export PYTHONPATH=.
   streamlit run streamlit_app.py
   ```

   Open [http://localhost:8501](http://localhost:8501).

## Docker (app + database)

Compose injects `DATABASE_URL` for the `app` service. You still need GCP credentials **inside** the container, for example:

- Mount a service account JSON and set `GOOGLE_APPLICATION_CREDENTIALS` to the path inside the container, **or**
- Run the app on a platform that attaches a workload identity (GKE / Cloud Run).

Example pattern (adjust paths):

```yaml
# add under app.volumes in docker-compose.yml after you have a key file:
# volumes:
#   - ./secrets/sa.json:/run/secrets/sa.json:ro
# and in environment:
#   GOOGLE_APPLICATION_CREDENTIALS: /run/secrets/sa.json
```

Then:

```bash
docker compose up --build
```

## Langfuse

Create a project in [Langfuse Cloud](https://cloud.langfuse.com) (or self-host), then set in `.env`:

- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_HOST` (e.g. `https://us.cloud.langfuse.com` for US data region)

If keys are empty, tracing is skipped and the app still runs.

## SonarQube

- Project metadata is in `sonar-project.properties` (update `sonar.projectKey` to match your SonarQube / SonarCloud project).
- Run analysis with your standard pipeline or locally using `sonar-scanner`, with `sonar.host.url` and authentication token supplied by your environment (do not commit tokens).

## Layout

| Path | Purpose |
|------|---------|
| `app/config.py` | Env-based settings (GCP, DB, Langfuse) |
| `app/db.py` | SQLAlchemy + PostgreSQL helper |
| `app/llm.py` | Gemini / Vertex generation + optional Langfuse |
| `app/observability.py` | Langfuse client factory |
| `streamlit_app.py` | Streamlit chat shell |
| `docker-compose.yml` | `db` + `app` services |
| `Dockerfile` | Production-style app image |

## Model ID

Default model is **`gemini-2.0-flash`** (`GEMINI_MODEL`). If Vertex returns “model not found” for your region, check the [Vertex AI model list](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models) and set `GEMINI_MODEL` to the exact ID supported in `GOOGLE_CLOUD_LOCATION`.
