"""Prompt templates for Stage 2 perception and reasoning."""

from __future__ import annotations

KNOWN_OBJECT_CATEGORIES = [
    "Clock",
    "Teddy Bear",
    "Bowl",
    "Apple",
    "Banana",
    "Vase",
    "Cup",
    "Cake",
    "Bottle",
    "Book",
]

LEVEL1_TASK_PROMPT = """Analyze the attached photo of the laptop screen displaying task instructions.
The task instruction is typically in the format of:
"Please go to Point X and identify the object there."
There will be two points we need to visit.

Known locations in the venue: {known_locations}.

Extract and return the names of the two points we need to visit.
"""

CLASSIFY_OBJECT_PROMPT = """Analyze the attached photo of an object placed at Point {point}.
Your task is to identify which category the object belongs to.

Known object categories:
{categories_list}
*(Note: unexpected objects may occasionally appear, but prefer matching to the known set if appropriate).*

Identify the object category and primary color.
"""

LEVEL2_TASK_PROMPT = """Analyze the attached photo of the laptop screen displaying task instructions.
The task instruction is typically in the format of:
"Please find the point where X is located."
There will be two objects we need to find.

Known object categories:
{categories_list}

Extract and return the categories of the two objects we need to search for.
"""

LEVEL3_TASK_INFO_PROMPT = """Analyze the attached photo of the laptop screen displaying Level 3 task instructions.

Your task:
1. Transcribe the task instruction verbatim.
2. Determine if completing this task requires physical size, height, width, volume, or dimension measurements
   (such as "Where is the tallest object in the venue?", "Find the shortest object", "Measure the height of...", "Which object is largest?").
   - Set use_high_visibility = True if measuring physical dimensions or finding tallest/shortest is required.
   - Set use_high_visibility = False if the task only asks about visual appearance, colors, or item counts
     (such as "Please count how many objects in the venue have the same color as the object at point D").
"""

LEVEL3_SOLVE_PROMPT_TEMPLATE = """You are the perception and reasoning module of an autonomous TurtleBot4 Lite robot.

### CONTEXT & TASK:
Task Instruction displayed on screen:
"{task_description}"

The robot has visited the following locations:
- Point 0: Task Center (laptop screen displaying task instructions)
- Inspected Object Locations: {inspected_locations}

### REFERENCE INFORMATION:
- Reference height: The pedestal/box the objects sit upon is 9.0 cm high.
- Known possible objects (roughly from smallest to tallest):
  1. Apple (~7-9 cm)
  2. Cake
  3. Mug / Cup (~9-12 cm)
  4. Alarm clock (~10-15 cm)
  5. Banana (~15-20 cm)
  6. Book
  7. Bottle 500ml (~20-23 cm)
  8. Teddy Bear (sitting vs. standing height varies; typically taller than a 500ml bottle)
  9. Vase (~25-35 cm)
  10. Bottle 1.5L (~30-35 cm)

### PRELIMINARY ROBOT SENSOR ESTIMATES:
{preliminary_estimates}

### INSTRUCTIONS:
1. Cross-reference the sensor estimates with the visual photos for each location.
   If a preliminary sensor estimate appears noisy or incorrect, adjust it based on visual evidence.
2. Build an internal inventory listing each inspected point with its final object class, color, and height.
3. Provide the direct, accurate answer to the task instruction.
"""

