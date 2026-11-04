"""High-level LLM task workflows for Levels 1, 2, and 3."""

from __future__ import annotations

from typing import Optional

from .llm_api import LabeledImage, query_structured
from .prompts import (
    CLASSIFY_OBJECT_PROMPT,
    KNOWN_OBJECT_CATEGORIES,
    LEVEL1_TASK_PROMPT,
    LEVEL2_TASK_PROMPT,
    LEVEL3_LEGIBILITY_PROMPT,
    LEVEL3_SOLVE_PROMPT_TEMPLATE,
)
from .schemas import (
    Categories,
    Category,
    Level3Solution,
    TaskLegibility,
    Waypoints,
)


def parse_level1_task(task_photo_bytes: bytes, known_locations: list[str]) -> list[str]:
    """Parse Level 1 instructions photo and return list of point names to visit (e.g. ['B', 'E'])."""
    loc_str = ", ".join(sorted(known_locations))
    prompt = LEVEL1_TASK_PROMPT.format(known_locations=loc_str)
    images = [LabeledImage(label="photo_task_laptop", jpeg_bytes=task_photo_bytes)]

    res: Waypoints = query_structured(
        prompt=prompt,
        images=images,
        response_model=Waypoints,
    )
    # Normalize names to uppercase stripped
    return [wp.name.strip().upper() for wp in res.waypoints if wp.name.strip()]


def classify_object(photo_bytes: bytes, point_name: str) -> Category:
    """Identify the category, color, and estimated height of the object at point_name from a photo."""
    cat_str = "\n".join(f"- {c}" for c in KNOWN_OBJECT_CATEGORIES)
    prompt = CLASSIFY_OBJECT_PROMPT.format(point=point_name, categories_list=cat_str)
    images = [LabeledImage(label=f"photo_point_{point_name}", jpeg_bytes=photo_bytes)]

    res: Category = query_structured(
        prompt=prompt,
        images=images,
        response_model=Category,
    )
    return res


def parse_level2_task(task_photo_bytes: bytes) -> list[str]:
    """Parse Level 2 instructions photo and return target object categories (e.g. ['Mug', 'Book'])."""
    cat_str = "\n".join(f"- {c}" for c in KNOWN_OBJECT_CATEGORIES)
    prompt = LEVEL2_TASK_PROMPT.format(categories_list=cat_str)
    images = [LabeledImage(label="photo_task_laptop", jpeg_bytes=task_photo_bytes)]

    res: Categories = query_structured(
        prompt=prompt,
        images=images,
        response_model=Categories,
    )
    return [cat.category.strip() for cat in res.categories if cat.category.strip()]


def check_level3_legibility(task_photo_bytes: bytes) -> TaskLegibility:
    """Check asynchronously whether the photographed Level 3 question at Point 0 is clearly legible."""
    images = [LabeledImage(label="photo_task_laptop", jpeg_bytes=task_photo_bytes)]
    res: TaskLegibility = query_structured(
        prompt=LEVEL3_LEGIBILITY_PROMPT,
        images=images,
        response_model=TaskLegibility,
    )
    return res


def solve_level3(
    task_photo_bytes: bytes,
    point_photos: Optional[dict[str, bytes]] = None,
    sensor_estimates: Optional[dict[str, Optional[dict]]] = None,
    transcribed_task: Optional[str] = None,
) -> Level3Solution:
    """Synthesize complete Level 3 solution in one shot from task photo, point photos, and GPU estimates."""
    point_photos = point_photos or {}
    sensor_estimates = sensor_estimates or {}
    inspected_str = ", ".join(f"Point {p}" for p in sorted(point_photos.keys()))

    # Build GPU sensor estimate block
    est_lines = []
    for pt in sorted(point_photos.keys()):
        est = sensor_estimates.get(pt)
        if est and est.get("primary"):
            p_data = est["primary"]
            c_name = p_data.get("class_name", "unknown")
            conf = p_data.get("confidence")
            conf_str = f"conf {conf:.2f}" if isinstance(conf, (int, float)) else ""
            h_val = p_data.get("height_cm")
            h_str = f"height {h_val:.1f} cm" if h_val is not None else "height n/a"
            w_val = p_data.get("width_cm")
            w_str = f"width {w_val:.1f} cm" if w_val is not None else ""
            bbox = p_data.get("bbox_px")
            bbox_str = f"bbox {bbox}" if bbox else ""
            parts = [s for s in (c_name, conf_str, h_str, w_str, bbox_str) if s]
            est_lines.append(f"- Point {pt}: [" + ", ".join(parts) + "]")
        else:
            est_lines.append(f"- Point {pt}: [No GPU detection / stereo measurement available]")

    estimates_str = "\n".join(est_lines) if est_lines else "- None"

    if transcribed_task is not None and transcribed_task.strip():
        task_context = f'Task Instruction displayed on screen:\n"{transcribed_task.strip()}"'
    else:
        task_context = (
            "Task Instruction: Read and transcribe verbatim directly from the first attached image "
            "labeled 'photo_task_laptop' (captured at Point 0)."
        )

    prompt = LEVEL3_SOLVE_PROMPT_TEMPLATE.format(
        task_context=task_context,
        inspected_locations=inspected_str or "None",
        preliminary_estimates=estimates_str,
    )

    # Attach task photo followed by photos for each location
    images = [LabeledImage(label="photo_task_laptop", jpeg_bytes=task_photo_bytes)]
    for pt in sorted(point_photos.keys()):
        images.append(LabeledImage(label=f"photo_point_{pt}", jpeg_bytes=point_photos[pt]))

    res: Level3Solution = query_structured(
        prompt=prompt,
        images=images,
        response_model=Level3Solution,
    )
    return res

