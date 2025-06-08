# Task Description: Prompt-Tuning & Evaluation Platform for a Debugging Assistant

## 1. Objective  
Build a lightweight web platform that lets you **author**, **evaluate**, and **compare** multiple prompt templates for an LLM-based code-debugging assistant. Each prompt version is run on a fixed “basket” of debugging cases and **scored** automatically on configurable criteria via a secondary LLM. You can then **inspect failures**, iterate on prompts, and track prompt-version performance over time.

---

## 2. Scope & Key Features  

1. **Basket of Debug Cases**  
   - A JSON list of N entries, each containing:
     - `id`: unique integer  
     - `problem`: text description  
     - `program`: code snippet  
     - `error`: runtime/compiler message  

   ```json
   [
     {
       "id": 1,
       "problem": "What is causing the TypeError?",
       "program": "def foo(x): return x + 1\nfoo('a')",
       "error": "TypeError: can only concatenate str (not \"int\") to str"
     },
     …
   ]
   ```

2. **Prompt Templates**  
   - Jinja2-style templates to interpolate each case’s fields into a “fix-generation” prompt.  
   - Users submit new templates via an **HTML form**.

3. **Single, Multi-Tag Evaluation Prompt**  
   - One configurable template that asks the evaluator LLM to judge **all** criteria in a single response, using distinct XML-like tags, e.g.:

     ```xml
     <correctness>Yes</correctness>
     <code_presence>No</code_presence>
     <line_reference>Yes</line_reference>
     ```

   - Criteria tags are fully configurable (add/remove/edit).

4. **Configurable Metrics**  
   - Default criteria:  
     1. **correctness** – Does the fix resolve the error?  
     2. **code_presence** – Does the reply include code?  
     3. **line_reference** – Is any referenced line number accurate?  
   - Add new metrics by extending the evaluation-prompt and parsing logic.

5. **Dependency-Aware Pipeline**  
   - **Stage 1**: **Generate Answers** (parallel per case)  
   - **Stage 2**: **Evaluate Answers** (depends on generation)  
   - **Stage 3**: **Parse & Compute Flags** (depends on evaluation)  
   - **Stage 4**: **Aggregate Metrics** (depends on all cases)

6. **Versioning & Storage**  
   - **`prompt_versions`** (one record per template):
     ```json
     {
       "prompt_id": "uuid-v4",
       "prompt_text": "…Jinja2 template…",
       "created_at": "2025-06-08T10:12:00+04:00",
       "metrics": {
         "correctness": 0.85,
         "code_presence": 0.92,
         "line_reference": 0.78,
         "final_score": 0.75
       }
     }
     ```
   - **`responses`** (one row per case × prompt):
     ```json
     {
       "prompt_id": "uuid-v4",
       "case_id": 42,
       "answer": "…generated fix…",
       "raw_evaluation": "<correctness>Yes</…>",
       "flags": {
         "correctness": true,
         "code_presence": false,
         "line_reference": true
       },
       "final_flag": false
     }
     ```

7. **REST API**  
   - `POST /prompts`  
     - **Body**: `{ "prompt_text": "…Jinja2 template…" }`  
     - **Response**: `{ "prompt_id": "...", "status": "queued" }`  
     - **Side effect**: starts background pipeline  
   - `GET /prompts`  
     - Returns an array of all prompt versions, sorted by `metrics.final_score` desc  
   - `GET /prompts/{prompt_id}/worst`  
     - Returns responses with `final_flag == false` (or bottom K)

8. **Frontend UI**  
   - **Prompt submission form** (textarea + button + spinner)  
   - **Ranked prompt table** showing:
     | Rank | Prompt ID | Created At       | Correctness | Code % | Line Ref % | Final % |
     |------|-----------|------------------|-------------|--------|------------|---------|
     | 1    | uuid-v1   | 2025-06-08 10:15 | 92.5%       | 87.5%  | 78.3%      | 81.4%   |

   - **Details modal** listing worst‐performing cases per prompt.

9. **Concurrency & Orchestration**  
   - **Option A:** `asyncio` in-process  
   - **Option B:** Celery/RQ + Redis (chained tasks)  
   - **Option C:** Prefect/Airflow DAG  

10. **Tech Stack**  
    - **Backend**: Python 3.10+, FastAPI, Jinja2, `openai` SDK, SQLite (SQLModel)  
    - **Frontend**: HTML/CSS + vanilla JS (Fetch API)  
    - **Deployment**: Uvicorn ASGI, Docker (optional)  

---

## 3. Implementation Roadmap

1. **Data Layer**  
   - Define `basket.json` schema  
   - Implement storage for `prompt_versions` & `responses`

2. **Core Pipeline**  
   - Jinja2 rendering for fix generation  
   - Single, multi-tag evaluation prompt + parser  
   - Async tasks for generation → evaluation → parsing → aggregation

3. **API Endpoints**  
   - `POST /prompts` → enqueue pipeline  
   - `GET /prompts` → list versions + metrics  
   - `GET /prompts/{id}/worst` → fetch failed cases

4. **UI Prototype**  
   - HTML form + JS for submissions & table rendering  
   - Modal for worst-case details

5. **Orchestration**  
   - Start with `asyncio`; evolve to Celery or a DAG runner as needed

6. **Iterate & Test**  
   - Run on N ≈ 238 cases  
   - Verify metric parsing and UI interactions  
   - Tweak evaluation prompt tags and template versions

---

Deliver a **minimal, end-to-end prototype** covering the full cycle:

> **Author prompt → Generate fixes → Multi-tag evaluation → Aggregate scores → Store & display versions → Inspect failures → Repeat**  
