from fastapi import FastAPI, Depends, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.staticfiles import StaticFiles
from sqlmodel import select, delete
from sqlmodel.ext.asyncio.session import AsyncSession
from app.database import init_db, get_session
from app.model import PromptVersion, ResponseRecord, LLMInteraction
import uuid
import asyncio
import sys
from typing import List
from pipeline import process_prompt

# Ensure console output uses UTF-8 encoding when printing logs
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def on_startup():
    await init_db()

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.post("/prompts")
async def add_prompt(payload: dict, background: BackgroundTasks, session: AsyncSession = Depends(get_session)):
    text = payload.get("prompt_text", "")
    pid = str(uuid.uuid4())
    pv = PromptVersion(
        prompt_id=pid,
        prompt_text=text,
        metrics={
            "right_error_description": 0,
            "correct_hint": 0,
            "correct_or_absent_code": 0,
            "correct_line_reference": 0,
            "final_score": 0,
        },
    )
    session.add(pv)
    await session.commit()
    background.add_task(process_prompt, pid, text)
    return {"prompt_id": pid, "status": "queued"}

@app.get("/prompts")
async def list_prompts(session: AsyncSession = Depends(get_session)) -> List[PromptVersion]:
    result = await session.exec(select(PromptVersion))
    prompts = result.all()
    prompts.sort(key=lambda p: p.metrics.get("final_score", 0), reverse=True)
    return prompts

@app.get("/prompts/{prompt_id}/worst")
async def worst_cases(prompt_id: str, session: AsyncSession = Depends(get_session)) -> List[ResponseRecord]:
    result = await session.exec(select(ResponseRecord).where(ResponseRecord.prompt_id == prompt_id, ResponseRecord.final_flag == False))
    return result.all()

@app.delete("/prompts/{prompt_id}")
async def delete_prompt(prompt_id: str, session: AsyncSession = Depends(get_session)):
    await session.exec(delete(ResponseRecord).where(ResponseRecord.prompt_id == prompt_id))
    await session.exec(delete(PromptVersion).where(PromptVersion.prompt_id == prompt_id))
    await session.commit()
    return {"status": "deleted"}


@app.get("/logs")
async def download_logs(session: AsyncSession = Depends(get_session)):
    result = await session.exec(select(LLMInteraction))
    logs = result.all()
    data = jsonable_encoder(logs)
    headers = {"Content-Disposition": "attachment; filename=logs.json"}
    return JSONResponse(content=data, headers=headers)
