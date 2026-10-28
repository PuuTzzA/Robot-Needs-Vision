"""LLM API Interface and Stub Adapter for Stage 2 System.

================================================================================
INSTRUCTIONS FOR INTEGRATION
================================================================================
This module defines the abstract interface that connects Windows Mission Control
to a multimodal Large Language Model (e.g. Google Gemini 2.5/3.8 Flash, OpenAI GPT-4o,
or Anthropic Claude 3.5 Sonnet).

To connect a concrete LLM provider:
1. Open this file and locate `query_structured()`.
2. Replace the NotImplementedError stub with your provider SDK call.
3. Pass `response_model.model_json_schema()` to request structured JSON output.
4. Validate the response text using `response_model.model_validate_json(raw_json)`.

Example Implementation for Google Gemini (`pip install google-genai`):
----------------------------------------------------------------------
```python
from google import genai
from google.genai import types

_gemini_client = None

def get_client():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client() # picks up GEMINI_API_KEY from environment
    return _gemini_client

def query_structured(prompt, images, response_model, system=None, timeout_s=60.0):
    client = get_client()
    contents = []
    if system:
        contents.append(f"System: {system}\n")
    for img in images:
        contents.append(f"[{img.label}]:")
        contents.append(types.Part.from_bytes(data=img.jpeg_bytes, mime_type=img.mime_type))
    contents.append(prompt)

    response = client.models.generate_content(
        model="gemini-2.5-flash",  # or gemini-3.8-flash
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_model,
        ),
    )
    return response_model.model_validate_json(response.text)
```
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Optional, Type, TypeVar
from pydantic import BaseModel

from .schemas import (
    Categories,
    Category,
    InventoryItem,
    Level3Solution,
    Level3TaskInfo,
    Waypoint,
    Waypoints,
)

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when an LLM query fails or returns unparseable output."""
    pass


@dataclass
class LabeledImage:
    """An image attached to an LLM prompt with a semantic label."""
    label: str  # e.g. "photo_task_laptop", "photo_point_B"
    jpeg_bytes: bytes
    mime_type: str = "image/jpeg"


# Global flag to enable mock simulation for testing/offline runs
MOCK_MODE_ENABLED: bool = False


def set_mock_mode(enabled: bool) -> None:
    """Toggle offline Mock LLM for integration testing without live API keys."""
    global MOCK_MODE_ENABLED
    MOCK_MODE_ENABLED = enabled


def is_mock_mode() -> bool:
    return MOCK_MODE_ENABLED


def health_check() -> tuple[bool, str]:
    """Test connectivity to the LLM backend. Used by 'Ping LLM' command."""
    if MOCK_MODE_ENABLED:
        return True, "Mock LLM active (offline simulation)"
    # When integrating a real provider, perform a lightweight check here
    return False, "LLM stub not configured. Run with --mock-llm or implement llm_api.py"


def query_structured(
    prompt: str,
    images: list[LabeledImage],
    response_model: Type[T],
    *,
    system: Optional[str] = None,
    timeout_s: float = 60.0,
) -> T:
    """Execute a multimodal structured query against the configured LLM.

    Args:
        prompt: User instructions describing the query and expectations.
        images: List of labeled JPEG images to provide as visual context.
        response_model: The Pydantic model class defining expected output schema.
        system: Optional system instruction / role definition.
        timeout_s: Timeout in seconds for the network call.

    Returns:
        An instance of `response_model` validated from the LLM output.

    Raises:
        LLMError: If communication or schema validation fails.
        NotImplementedError: If called while running as an unconfigured stub.
    """
    if MOCK_MODE_ENABLED:
        return _mock_query_structured(prompt, images, response_model)

    # -------------------------------------------------------------------------
    # STUB IMPLEMENTATION
    # -------------------------------------------------------------------------
    raise NotImplementedError(
        "query_structured() is currently a STUB.\n"
        "To enable real queries: edit Stage2/stage2_system/laptop/llm/llm_api.py "
        "and implement your preferred SDK (Gemini, OpenAI, Claude).\n"
        "To run in offline simulation mode: start the application with --mock-llm."
    )


# -----------------------------------------------------------------------------
# Offline Mock Implementation (for automated unit and integration tests)
# -----------------------------------------------------------------------------

def _mock_query_structured(
    prompt: str,
    images: list[LabeledImage],
    response_model: Type[T],
) -> T:
    """Returns plausible mock responses matching the required Pydantic schemas."""
    p_lower = prompt.lower()

    # 1. Level 1 Task -> return 2 waypoints
    if response_model is Waypoints:
        return Waypoints(waypoints=[Waypoint(name="B"), Waypoint(name="E")])  # type: ignore

    # 2. Level 2 Task -> return 2 categories
    if response_model is Categories:
        return Categories(categories=[Category(category="Cup"), Category(category="Book")])  # type: ignore

    # 3. Object Classification (Level 1 / Level 2 per point)
    if response_model is Category:
        # Infer from label if present, or assign mock category
        for img in images:
            if "point_b" in img.label.lower():
                return Category(category="Cup", color="white", confidence=0.92)  # type: ignore
            elif "point_e" in img.label.lower():
                return Category(category="Banana", color="yellow", confidence=0.88)  # type: ignore
            elif "point_d" in img.label.lower():
                return Category(category="Bottle", color="blue", confidence=0.95)  # type: ignore
            elif "point_c" in img.label.lower():
                return Category(category="Apple", color="red", confidence=0.90)  # type: ignore
        return Category(category="Mug", color="blue", confidence=0.85)  # type: ignore

    # 4. Level 3 Task Info
    if response_model is Level3TaskInfo:
        # Check if prompt or mock test mentions height
        is_height_task = "tallest" in p_lower or "height" in p_lower or "measure" in p_lower
        desc = (
            "Where is the tallest object in the venue?"
            if is_height_task
            else "Please count how many objects in the venue have the same color as the object at point D."
        )
        return Level3TaskInfo(task_description=desc, use_high_visibility=is_height_task)  # type: ignore

    # 5. Level 3 Final Solution
    if response_model is Level3Solution:
        inventory = [
            InventoryItem(point="B", category="Cup", color="white", height_cm=10.5),
            InventoryItem(point="C", category="Apple", color="red", height_cm=8.0),
            InventoryItem(point="D", category="Bottle", color="blue", height_cm=31.5),
            InventoryItem(point="E", category="Banana", color="yellow", height_cm=16.0),
        ]
        if "color" in p_lower:
            ans = "There are 2 objects with the same color as Point D (blue): Point D and Point H."
        else:
            ans = "The tallest object in the venue is the 1.5L Bottle at Point D with a measured height of 31.5 cm."

        return Level3Solution(  # type: ignore
            task_description="Evaluated from inspected locations.",
            inventory=inventory,
            answer=ans,
            reasoning="Compared visual evidence and stereo depth measurements against known 9cm box base reference.",
        )

    raise ValueError(f"MockLLM does not have a canned response for schema: {response_model}")

