"""Pydantic models: the typed contracts between the LLM and your code.

A schema turns "the model returned some text" into "the model returned an object
my code can check". If validation fails, you retry or stop; you never pass bad data on.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    # extra="forbid" -> JSON schema gets additionalProperties: false (required by strict tool use)
    model_config = ConfigDict(extra="forbid")


# ---------- tool inputs (the LLM fills these when it calls a tool) ----------


class SearchDocsInput(Strict):
    query: str = Field(description="What to look for in the maintenance documents.")
    k: int = Field(default=4, ge=1, le=10, description="How many passages to return.")


class AnalyzeSensorInput(Strict):
    asset_id: Literal["T1", "T2", "P-101"] = Field(description="Asset to analyse.")
    signal: Literal["bearing_temp_c", "vibration_mm_s"] = Field(description="Sensor signal.")
    days: int = Field(default=14, ge=3, le=60, description="Recent window in days.")


class WorkOrdersInput(Strict):
    asset_id: Literal["T1", "T2", "P-101"]


class CreateWorkOrderInput(Strict):
    asset_id: Literal["T1", "T2", "P-101"]
    description: str
    priority: Literal["low", "medium", "high"]


# ---------- structured outputs ----------


class TicketTriage(Strict):
    """Used in script 02: classify a short operator message."""

    asset_id: Literal["T1", "T2", "P-101", "unknown"]
    category: Literal["temperature", "vibration", "leak", "electrical", "other"]
    urgency: Literal["low", "medium", "high"]
    needs_human: bool
    reason: str = Field(description="One sentence.")


class Finding(Strict):
    statement: str = Field(description="One factual sentence.")
    citations: list[str] = Field(
        description="IDs of the sources that support it: chunk IDs like 'MAN-T-01#2' "
        "or tool results like 'tool:analyze_sensor:T2:bearing_temp_c'."
    )


class Report(Strict):
    """The final output of the Turbine Assistant."""

    asset_id: Literal["T1", "T2", "P-101", "unknown"]
    summary: str = Field(description="Two or three sentences for the engineer.")
    findings: list[Finding]
    recommended_actions: list[str]
    needs_work_order: bool
    confidence: float = Field(ge=0.0, le=1.0, description="Probability the summary is correct.")
    unknowns: list[str] = Field(description="What the assistant could not check.")
    ai_generated: bool = Field(default=True, description="Always true: responsible-AI label.")


class JudgeVerdict(Strict):
    """Used by the LLM-as-judge in script 14."""

    passed: bool
    score: float = Field(ge=0.0, le=1.0, description="Judge confidence that the answer is correct.")
    reason: str
