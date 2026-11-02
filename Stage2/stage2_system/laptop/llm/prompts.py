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

LEVEL3_LEGIBILITY_PROMPT = """Analyze the attached photo of the laptop screen displaying the Level 3 task question.

Determine whether the task sentence/question on the laptop screen is completely legible:
- Set `legible = True` ONLY if the screen is visible and the entire sentence/question can be read clearly without guessing any words, object names, point letters, or numbers.
- Set `legible = False` if the screen is missing, cut off, blurry, obscured, washed out by glare, or if any part of the question is unreadable.
"""

LEVEL3_SOLVE_PROMPT_TEMPLATE = """You are the perception and reasoning module of an autonomous TurtleBot4 Lite robot.

### CONTEXT & TASK:
{task_context}

The robot has visited the following locations:
- Point 0: Task Center (laptop screen displaying task instructions — attached as 'photo_task_laptop')
- Inspected Object Locations: {inspected_locations}

### REFERENCE INFORMATION:
- Reference height: The pedestal/box the objects sit upon is 7.0 cm high.
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

### PRELIMINARY GPU SERVER STEREO & YOLO ESTIMATES (Note: The preliminary estimates are very inacurate and YOLO often detects the incorrect object, e.g. a TV (which is not a possible object) in the background. Trust your own reasoning over the estimates from YOLO (the GPU Server)):
{preliminary_estimates}

### INSTRUCTIONS:
1. Read 'photo_task_laptop' carefully and record the verbatim task instruction in `task_description`.
2. Cross-reference the GPU server estimates (class, bounding box, stereo height) with the visual photos for each location.
   If a preliminary GPU estimate appears noisy, misclassified, or missing, refine it using visual evidence from the photo.
3. Build an inventory listing each inspected point with its final object class, color, and height.
4. Provide the answer in `answer` as a complete, informative sentence — NEVER reply with only a bare number, single word, or bare point letter.
   Always state the direct answer AND explicitly name the relevant objects and their point locations in a full sentence.
   - Example (counting/comparison): If the prompt is "How many objects are taller than the object at point D?", do NOT just answer "3"; instead answer:
     "3. The Vase at point B, the Clock at point C and the Book at point I are taller than the Cake at point D."
   - Example (color count): "2. The Bottle at point B and the Cup at point H have the same blue color as the Bowl at point D."
   - Example (superlative/location): "Point D. The Bottle at point D is the tallest object in the venue with an estimated height of 31.5 cm."
"""

