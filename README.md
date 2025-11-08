# Ultimate RAG Analytics Assistant

The Ultimate RAG Analytics Assistant is an end-to-end Retrieval Augmented Generation (RAG) system for business analytics. It combines Google BigQuery, Vertex AI (for Gemini text and embedding models), Qdrant vector search, and a Flask API to answer ad-hoc analytical questions, generate SQL, surface insights, and persist conversation context. A lightweight web front-end provides a chat-style experience.

---

## Highlights
- Multi-stage hybrid retrieval that mixes keyword, embedding, knowledge graph, and feedback signals.
- Advanced SQL planner/validator to safely query BigQuery tables and run dry-run cost checks.
- Built-in analytics layer to summarize results, generate insights, charts, and recommendations.
- Conversation memory with chart persistence, dashboard assembly, and feedback learning hooks.
- Flask API with REST endpoints for asking questions, intent classification, query decomposition, insight generation, chart storage, and feedback capture.
- Optional single-page front-end (`frontend/index.html`) styled as “BusinessGPT”.

---

## Project Structure
- `ultimate_rag_pipeline.py` – Core pipeline: retrieval, understanding, SQL generation, analytics, memory, and learning systems.
- `ultimate_rag_api.py` – Flask application exposing pipeline capabilities over HTTP.
- `run_ultimate_rag.py` – Interactive launcher to check dependencies, run tests, and start the API server.
- `test_ultimate_rag.py` – Smoke tests for core components and end-to-end question handling.
- `requirements.txt` / `requirements_ultimate.txt` – Python dependencies (pin to `requirements_ultimate.txt` for full feature set).
- `frontend/` – Static HTML/CSS/JS chat interface.
- `credentials/` – Placeholder for Google Cloud service account JSON files (ignored by VCS).
- `old_project/` – Legacy scripts retained for reference.
- `RAG_COMPARISON.md`, `README_Ultimate_RAG.md` – Historical documentation.

---

## Prerequisites
- Python 3.10 or newer.
- Access to Google Cloud Project(s) with:
  - BigQuery datasets referenced in the pipeline (defaults: project `skf-bq-analytics-hub`, dataset `mrep_skf`).
  - Vertex AI enabled with Gemini text and embedding models (defaults: project `tcl-vertex-ai`, location `asia-southeast1`).
- Running Qdrant instance (local Docker or managed). Defaults expect `http://localhost:6333`.
- Service account JSON credentials for BigQuery and Vertex AI placed in `credentials/` (filenames referenced in `ultimate_rag_pipeline.py`).
- (Optional) Google Charts-capable browser to run the included front-end.

---

## Setup
1. **Create a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # macOS / Linux
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements_ultimate.txt
   ```
3. **Configure credentials**
   - Place BigQuery service account JSON in `credentials/bm-409@skf-bq-analytics-hub.iam.gserviceaccount.com.json`.
   - Place Vertex AI service account JSON in `credentials/tcl-vertex-ai-58aa168440df.json`.
   - Update the constants near the top of `ultimate_rag_pipeline.py` if your filenames, projects, or locations differ.
   - (Optional) export `GOOGLE_APPLICATION_CREDENTIALS` pointing to each file if you prefer environment-based authentication.
4. **Start Qdrant**
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```
5. **Prime BigQuery/Qdrant connections**
   - Ensure the referenced tables (default: `sales_details`) exist in BigQuery.
   - The pipeline will re-create the Qdrant collection (`pharma_ultimate_embeddings`) on startup.

---

## Running Locally
1. **Interactive launcher**
   ```bash
   python run_ultimate_rag.py
   ```
   - Option 1: run the smoke tests.
   - Option 2: start the API server (`http://localhost:5000`).
   - Option 3: run tests then start the server.
2. **Direct API run**
   ```bash
   python ultimate_rag_api.py
   ```
3. **Frontend**
   - Open `frontend/index.html` in a browser (served via `file://` or any static hosting).
   - Update the API base URL in the script block if you deploy the backend elsewhere.

---

## Key API Endpoints
All routes are prefixed by the Flask server root (default `http://localhost:5000`).

- `POST /ask` – Main chat endpoint. Payload: `{"question": "...", "user_id": "optional", "session_id": "optional"}`. Returns summary, generated SQL, tabular results, charts, insights, metadata, and citation hints.
- `POST /intent` – Classify query intent. Payload: `{"query": "..."}`. Returns intent type, confidence, entities, and temporal context.
- `POST /decompose` – Break complex question into sub-queries.
- `POST /insights` – Generate additional narrative insights from supplied results payloads.
- `POST /charts/save` – Persist charts for later dashboard creation.
- `GET /charts/list/<user_id>/<session_id>` – List saved charts.
- `POST /dashboard/create` – Build a dashboard from selected chart IDs.
- `GET /dashboards/list/<user_id>/<session_id>` – Enumerate dashboards per session.
- `POST /feedback` – Submit user feedback for learning loops.
- `GET /conversation/<user_id>/<session_id>` – Retrieve conversation history and preferences.
- `GET /health` – Health status with enabled features.

Refer to `ultimate_rag_api.py` for additional helpers and response schemas.

---

## Testing
```bash
python test_ultimate_rag.py
```

The test suite exercises:
- Intent classification, retrieval, knowledge graph connections, analytics outputs.
- End-to-end query execution through `UltimateRAGPipeline.ask_intelligent`.
- Basic sanity checks for chart/insight generation.

> **Note:** Tests expect live services (Qdrant, BigQuery, Vertex AI). Use mocks or disable sections if running offline.

---

## Configuration Tips
- Adjust constants in `ultimate_rag_pipeline.py` to point to your datasets, models, and cache filenames.
- Cache files such as `conversation_cache.json`, `keyword_cache.json`, and `knowledge_graph.pkl` are generated at runtime; they are excluded from version control via `.ignore`.
- To change top-k retrieval counts, scoring weights, or SQL safety thresholds, inspect the respective classes (e.g., `EnhancedMultiStageRetriever`, `AdvancedSQLValidator`, `AdvancedStatisticalAnalyzer`).
- The front-end consumes metadata fields like `charts`, `insights`, `citations`, and `dry_run`. Ensure any custom responses keep those keys to avoid breakage.

---

## Deployment Notes
- Containerize with Gunicorn/Uvicorn or Cloud Run for production; ensure Vertex AI and BigQuery credentials are mounted securely.
- Harden the Flask server (rate limits, authentication, CORS restrictions) before exposing publicly.
- Consider storing conversation memory, saved charts, and dashboards in persistent storage rather than local JSON files for multi-instance deployments.

---

## Troubleshooting
- **Missing dependencies** – Run `python run_ultimate_rag.py` and choose option 1; the launcher lists missing packages.
- **Qdrant connection errors** – Confirm the container is running and accessible. Update connection parameters in `ultimate_rag_pipeline.py` if using managed Qdrant.
- **Authentication errors** – Verify service account JSON contents and IAM permissions (BigQuery Data Viewer/Job User, Vertex AI User).
- **High latency** – Gemini and BigQuery calls can be slow; review `AdvancedQueryProcessor` batching, enable BigQuery dry runs, or cache frequent queries.
- **Model quota** – Monitor Vertex AI usage; fall back to deterministic heuristics when quota is exceeded (the pipeline already implements some fallbacks).

---

## Security
- Keep the `credentials/` directory out of version control (enforced by `.ignore`) and restrict file permissions.
- Rotate service account keys periodically.
- Pre-filter user questions or add allowlists to avoid executing unexpected SQL against BigQuery.

---

## Roadmap Ideas
- Swap Gemini models via environment variables instead of hard-coded names.
- Add structured logging and tracing (e.g., OpenTelemetry).
- Persist chart/dashboard metadata in a database for collaborative usage.
- Automate schema embedding refresh jobs for new tables.

---

Happy analyzing! 🎯

