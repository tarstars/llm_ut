# MVP Development Plan

This document outlines the steps required to build a minimal viable product for the prompt-tuning and evaluation platform described in `task_description.md`.

## 1. Environment Setup

1. Install dependencies using `install_dependencies` script.
2. Prepare an `.env` file with required OpenAI credentials.
3. Initialize the SQLite database using the utility in `app/database.py`.

## 2. Data Layer

- Define the structure of `basket.json` and include enough cases for testing.
- Create the SQLModel tables (`PromptVersion` and `ResponseRecord`) by calling `init_db()`.

## 3. Pipeline

1. Implement `pipeline.py` with four sequential stages:
   - **Generate Answers**: Render each prompt template for every basket case and call OpenAI to get an answer.
   - **Evaluate Answers**: Feed generated answers to the evaluation prompt and capture the raw response.
   - **Parse Flags**: Extract metric flags from the raw evaluation using the configured XML-like tags.
   - **Aggregate Metrics**: Compute summary metrics for each prompt version and store them.
2. Use `asyncio` to run stage 1 and 2 concurrently per case.

## 4. API Endpoints

Build a small FastAPI application with the following endpoints:

- `POST /prompts` — submit a new prompt template and start the pipeline.
- `GET /prompts` — list prompt versions sorted by `final_score`.
- `GET /prompts/{prompt_id}/worst` — return responses with failing `final_flag` values.

## 5. Frontend Prototype

Create a static HTML/JS interface (see `static/index.html` and `static/app.js`) with:

- A form to submit a prompt template.
- A table showing ranked prompt versions and metrics.
- A modal dialog to display the worst-performing cases for a selected prompt.

## 6. Orchestration and Testing

- Start with an in-process `asyncio` pipeline.
- Provide a script to run the pipeline manually for local testing.
- Plan for later migration to Celery or another task runner if concurrency demands grow.

## 7. Iteration and Next Steps

- Add more evaluation metrics and adjust the evaluation prompt template.
- Increase the number of basket cases and refine examples.
- Polish the UI and consider Docker for deployment.

This plan delivers an end-to-end MVP that lets users submit prompts, run evaluations, and review results through a simple web interface.
