"""Pydantic schemas for structured LLM inputs and outputs."""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class Waypoint(BaseModel):
    name: str = Field(description="Name or ID of the waypoint (e.g. 'B', 'C').")


class Waypoints(BaseModel):
    waypoints: List[Waypoint] = Field(description="List of waypoints.")


class Category(BaseModel):
    category: str = Field(description="Object category name.")
    color: Optional[str] = Field(default=None, description="Primary color of the object, if identifiable.")
    confidence: Optional[float] = Field(default=None, description="Model confidence score between 0.0 and 1.0.")


class Categories(BaseModel):
    categories: List[Category] = Field(description="List of object categories.")


class Level3TaskInfo(BaseModel):
    """Initial Level 3 task parsing response."""
    task_description: str = Field(description="Verbatim transcription of the instruction displayed on the screen.")
    use_high_visibility: bool = Field(
        description="True if the task requires size, height, width, volume, or dimension measurements (e.g. 'tallest', 'height', 'measure', 'size'). False for color-only or count-only tasks."
    )


class InventoryItem(BaseModel):
    point: str = Field(description="Location identifier (e.g. 'B', 'D').")
    category: str = Field(description="Final object category.")
    color: Optional[str] = Field(default=None, description="Identified object color.")
    height_cm: Optional[float] = Field(default=None, description="Measured or estimated height in cm.")


class Level3Solution(BaseModel):
    """Final Level 3 comprehensive solution."""
    task_description: str = Field(description="Verbatim task instruction.")
    inventory: List[InventoryItem] = Field(description="Inventory of all inspected object locations.")
    answer: str = Field(description="Direct, concise answer to the task instruction.")
    reasoning: Optional[str] = Field(default=None, description="Step-by-step reasoning explaining the answer.")

