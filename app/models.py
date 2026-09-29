"""Pydantic models = the contract of the API (validation + auto docs)."""
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000,
                          examples=["How many annual leave days do employees get?"])


class SourceItem(BaseModel):
    file: str
    chunk: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    mode: str = Field(description="'llm' when Claude generated the answer, "
                                  "'extractive' when no API key is set, "
                                  "'no_context' when nothing relevant was retrieved")
