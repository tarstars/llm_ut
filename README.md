# Prompt Tuning Evaluation Prototype

This project is a minimal implementation of the platform described in `task_description.md`.
It exposes a small FastAPI service that lets you submit Jinja prompt templates and view
aggregated evaluation results.

## Running

1. Install dependencies using `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: packages require internet access.*
2. Start the server:
   ```bash
   uvicorn main:app
   ```
3. Open `http://localhost:8000/` in a browser to use the dashboard.

By default generation and validation requests are sent to different LLM models.
Set the environment variable `USE_SINGLE_LLM_ENDPOINT=true` to send both
requests to the same endpoint.

To verify the code, run the test suite with:
```bash
pytest -q
```

The evaluation pipeline in `pipeline.py` is intentionally simple and does not call
any external LLMs. It renders the template against each case in `basket.json`,
performs a trivial evaluation, stores results in SQLite and updates metrics for
the prompt version.

All raw requests and responses exchanged with the models are stored in the
`LLMInteraction` table and can be downloaded via the `/logs` endpoint.
