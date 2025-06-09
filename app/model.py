from typing import Optional, Dict
from datetime import datetime
from sqlmodel import SQLModel, Field
from sqlalchemy import Column, JSON

class PromptVersion(SQLModel, table=True):
    prompt_id: str = Field(default=None, primary_key=True)
    prompt_text: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metrics: Dict[str, float] = Field(
        default_factory=dict, sa_column=Column(JSON)
    )  # e.g. correctness, code_presence, line_reference, final_score

class ResponseRecord(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    prompt_id: str = Field(foreign_key="promptversion.prompt_id")
    case_id: int
    answer: str
    raw_evaluation: str
    flags: Dict[str, bool] = Field(default_factory=dict, sa_column=Column(JSON))
    final_flag: bool


class LLMInteraction(SQLModel, table=True):
    """Store raw requests and responses sent to language models."""

    id: Optional[int] = Field(default=None, primary_key=True)
    prompt_id: str
    question_id: int
    request_type: str
    request_payload: Dict = Field(default_factory=dict, sa_column=Column(JSON))
    response: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
