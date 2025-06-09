# Prompt Tuning Evaluation Prototype

This project is a minimal implementation of the platform described in `task_description.md`.
It exposes a small FastAPI service that lets you submit Jinja prompt templates and view
aggregated evaluation results.

## Running

1. Install dependencies (FastAPI, SQLModel, Jinja2, Pytest, etc.):
   ```bash
   bash install_dependencies
   ```
   *Note: packages require internet access.*
2. Start the server:
   ```bash
   uvicorn main:app
   ```
3. Open `http://localhost:8000/` in a browser to use the dashboard.

To verify the code, run the test suite with:
```bash
pytest -q
```

The evaluation pipeline in `pipeline.py` is intentionally simple and does not call
any external LLMs. It renders the template against each case in `basket.json`,
performs a trivial evaluation, stores results in SQLite and updates metrics for
the prompt version.
