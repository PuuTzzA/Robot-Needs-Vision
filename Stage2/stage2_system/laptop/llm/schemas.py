"""Pydantic schemas for structured LLM inputs and outputs."""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class Waypoint(BaseModel):
    name: str = Field(description="Name or ID of the waypoint (e.g. 'B', 'C').")


class Waypoints(BaseModel):
    waypoints: List[Waypoint] = Field(description="List of waypoints.")


NO_OBJECT_SYNONYMS = frozenset(
    {"none", "no object", "no_object", "empty", "empty box", "nothing", "null", "n/a"}
)


def is_none_category(category: Optional[str]) -> bool:
    """Return True if a category string represents an empty box / no object present."""
    if category is None:
        return True
    cleaned = category.strip().lower()
    return not cleaned or cleaned in NO_OBJECT_SYNONYMS


class Category(BaseModel):
    category: str = Field(
        description=(
            "Object category name from the known categories, or 'None' if there is no object "
            "on top of the pedestal/box (the box is empty)."
        )
    )
    color: Optional[str] = Field(
        default=None,
        description="Primary color of the object if an object is present, or null if category is 'None'.",
    )
    height_cm: Optional[float] = Field(
        default=None,
        description="Estimated height of the object in cm (excluding any supporting pedestal/box), or null if category is 'None'.",
    )
    confidence: Optional[float] = Field(default=None, description="Model confidence score between 0.0 and 1.0.")

    @property
    def is_empty(self) -> bool:
        """True when no object is present on the pedestal/box."""
        return is_none_category(self.category)


class Categories(BaseModel):
    categories: List[Category] = Field(description="List of object categories.")


class TaskLegibility(BaseModel):
    """Legibility check response for the photographed Level 3 task screen."""
    legible: bool = Field(
        description=(
            "True if the task sentence/question on the laptop screen is completely visible and clearly readable "
            "without guessing. False if cut off, blurry, obscured, missing, or unreadable."
        )
    )
    transcribed_text: Optional[str] = Field(
        default=None,
        description="Verbatim transcription of the task question if legible, or None/partial text if illegible.",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Short reason if legible is False.",
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
    answer: str = Field(
        description=(
            "Complete full-sentence answer to the task instruction. Never output only a bare number or bare point "
            "letter; state the answer and explicitly name the relevant objects and their point locations in a full "
            "sentence (e.g. '3. The Vase at point B, the Clock at point C and the Book at point I are taller than "
            "the Cake at point D.')."
        )
    )
    reasoning: Optional[str] = Field(default=None, description="Step-by-step reasoning explaining the answer.")

