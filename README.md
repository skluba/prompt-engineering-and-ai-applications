# Conversational AI — data generation & NL querying

Python stack for a **Streamlit** app that calls **Gemini 2.0 Flash** on **Vertex AI** (via the **Google Gen AI SDK**), uses **PostgreSQL**, optional **Langfuse** tracing, and is containerized with **Docker**. **SonarQube** is configured via `sonar-project.properties`.

## Prerequisites

- **Python 3.11+**
- **Google Cloud** project with **Vertex AI API** enabled and billing (if required by your org)
- **Gemini on Vertex**: your principal needs a role such as **Vertex AI User** (`roles/aiplatform.user`)
- **Docker** + **Docker Compose v2.20+** (for `include` and the Langfuse stack)

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

## Phase 1: Synthetic data generation

The Streamlit app includes a **Data Generation** mode (sidebar) that:

1. Accepts a DDL file (`.sql`, `.txt`, or `.ddl`) or pasted DDL text — sample schemas live in the repo root (`company_employee_schema.ddl`, `library_mgm_schema.ddl`, `restrurants_schema.ddl`).
2. Lets you add free-text instructions, set **temperature**, and click **Generate** to produce JSON rows per table via **Gemini on Vertex**; results are validated for NOT NULL, PK/unique, ENUM literals, and foreign keys.
3. Shows a **preview per table**; each table has a feedback box and **Submit** to refine that table with the model.
4. **Save dataset to disk** writes CSVs + `manifest.json` under **`data/generated/<dataset_id>/`** (gitignored contents; the folder is kept via `data/generated/.gitkeep`). A **ZIP download** is offered from the same save action.

Use **Talk to your data** in the sidebar to browse saved datasets (CSV previews). Generic Gemini chat there does not yet query the CSVs; that is planned for a later phase.

## Docker (app + database)

Compose sets **`DATABASE_URL`** for the `app` service and, by default, bind-mounts **Application Default Credentials** from  
**`$HOME/.config/gcloud/application_default_credentials.json`** (after `gcloud auth application-default login`) read-only to  
**`/secrets/application_default_credentials.json`** in the container.

1. Run **`gcloud auth application-default login`** on the host if you have not already.
2. **Optional:** set **`GCP_ADC_HOST_PATH`** in `.env` to another host path (e.g. a service account JSON). If unset, Compose uses the gcloud ADC path above (compose expands **`${HOME}`** when you run `docker compose` from your shell).
3. **`GOOGLE_APPLICATION_CREDENTIALS`** in `docker-compose.yml` defaults to the in-container mount path.  
   In `.env`, either **omit** it or set it **only** to `/secrets/application_default_credentials.json`.  
   Do **not** set it to `/Users/...` — that is a host path and will not exist inside the Linux container.

4. If your `.env` still has an empty `GCP_ADC_HOST_PATH=` line from an old template, **delete that line** so the default applies (some setups treat an empty value as “set”).

Then:

```bash
docker compose up --build
```

For production, prefer a workload identity (GKE / Cloud Run) or a dedicated service account JSON mounted the same way.

## Langfuse

### Cloud

Create a project in [Langfuse Cloud](https://cloud.langfuse.com), then set in `.env`:

- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_HOST` (e.g. `https://cloud.langfuse.com` or `https://us.cloud.langfuse.com`)

### Self-hosted (this repo)

Docker Compose includes [`docker-compose.langfuse.yml`](docker-compose.langfuse.yml) (Langfuse **v3**: `lf-web`, `lf-worker`, dedicated Postgres, ClickHouse, Redis, MinIO). Requires **Docker Compose v2.20+** (`include`).

1. Start the stack: `docker compose up --build` (first boot can take a few minutes).
2. Open the UI at **http://localhost:3000**, create an organization and project, then **Project settings → API keys**. Copy the keys into `.env` as `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`.
3. The **app** container sets **`LANGFUSE_HOST=http://lf-web:3000`** in [`docker-compose.yml`](docker-compose.yml) so the Python SDK reaches Langfuse over the Docker network (this overrides `LANGFUSE_HOST` from `.env` for that service).
4. **Streamlit on the host** with Langfuse in Docker: use **`LANGFUSE_HOST=http://localhost:3000`** in `.env`.

Optional bootstrap: set `LANGFUSE_INIT_*` in `.env` per [Langfuse configuration](https://langfuse.com/docs/deployment/configuration). For non-dev use, replace **`LANGFUSE_ENCRYPTION_KEY`** (64 hex chars; e.g. `openssl rand -hex 32`), **`LANGFUSE_NEXTAUTH_SECRET`**, **`LANGFUSE_SALT`**, and MinIO/Redis/DB passwords.

**Ports:** Langfuse UI **3000**, MinIO S3 **9090**, MinIO console **127.0.0.1:9091**. The app database still publishes **5432**; Langfuse Postgres (`lf-postgres`) has **no** host port.

If your Compose build does not support `include`, run:  
`docker compose -f docker-compose.yml -f docker-compose.langfuse.yml up`.

If keys are empty, tracing is skipped and the app still runs.

**Tracing in this app** follows the [Langfuse instrumentation skill](.cursor/skills/langfuse/references/instrumentation.md): **sessions** (stable Streamlit `session_id`), optional **user id** from `st.user` when present, **tags** / **metadata** (`surface`, `model`), **model name** + **token usage** on the generation, **`flush()`** after each call, and trace **input** limited to the user message.

Optional `.env`: `LANGFUSE_RELEASE`, `LANGFUSE_TRACING_ENVIRONMENT`, `LANGFUSE_TRACE_VERSION`.

For deeper Langfuse + CLI + docs workflows in Cursor, use the bundled skill at [`.cursor/skills/langfuse/SKILL.md`](.cursor/skills/langfuse/SKILL.md) (from [langfuse/skills](https://github.com/langfuse/skills)).

## CI / QA (GitHub Actions)

Workflow: [`.github/workflows/qa.yml`](.github/workflows/qa.yml).

On each push / pull request to `main`, `master`, or `develop` it runs:

1. **Ruff** — lint and format check (`app`, `streamlit_app.py`, `tests`)
2. **pip-audit** — known vulnerabilities in `requirements.txt`
3. **Pytest + coverage** — tests with `coverage.xml` for SonarCloud
4. **SonarCloud** — static analysis and coverage upload (needs secrets below)
5. **Dependency review** — GitHub dependency review on pull requests (needs GitHub feature availability for your account)

Local parity:

```bash
pip install -r requirements.txt -r requirements-dev.txt
ruff check app streamlit_app.py tests && ruff format --check app streamlit_app.py tests
pytest --cov=app --cov-report=xml --cov-report=term-missing
```

## SonarCloud

This repo targets **[SonarCloud](https://sonarcloud.io)** with `sonar.organization=skluba` and  
`sonar.projectKey=skluba_prompt-engineering-and-ai-applications` in `sonar-project.properties`.

1. In SonarCloud, create or import the project so the **project key** matches `sonar-project.properties` exactly (adjust the file if the wizard assigns a different key).
2. In the GitHub repo, add **`SONAR_TOKEN`**: repository → **Settings → Secrets and variables → Actions** → **New repository secret**. Use a **new SonarCloud token** (Account → Security or project analysis token). **Do not commit tokens** or paste them in issues/PRs.
3. Push to GitHub; the QA workflow uploads analysis and coverage.

If you ever expose a token in chat or a commit, **revoke it in SonarCloud** and generate a new one.

## Layout

| Path | Purpose |
|------|---------|
| `app/config.py` | Env-based settings (GCP, DB, Langfuse) |
| `app/db.py` | SQLAlchemy + PostgreSQL helper |
| `app/tracing.py` | Langfuse trace context + GenAI usage mapping |
| `app/llm.py` | Gemini / Vertex generation + Langfuse generations |
| `app/schema_ddl.py` | DDL parsing for synthetic generation (CREATE TABLE, FKs) |
| `app/synthetic/` | Generate, validate, persist CSV/ZIP datasets under `data/generated/` |
| `app/observability.py` | Langfuse client factory |
| `.cursor/skills/langfuse/` | Langfuse Cursor skill (docs + CLI guidance) |
| `streamlit_app.py` | Streamlit: data generation + dataset browser + chat |
| `docker-compose.yml` | `db` + `app`; includes Langfuse stack |
| `docker-compose.langfuse.yml` | Self-hosted Langfuse v3 (`lf-*` services) |
| `Dockerfile` | Production-style app image |

## Model ID

Default model is **`gemini-2.0-flash`** (`GEMINI_MODEL`). If Vertex returns “model not found” for your region, check the [Vertex AI model list](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models) and set `GEMINI_MODEL` to the exact ID supported in `GOOGLE_CLOUD_LOCATION`.
