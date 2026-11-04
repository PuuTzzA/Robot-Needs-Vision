"""Prompt templates for Stage 2 perception and reasoning."""

from __future__ import annotations

KNOWN_OBJECT_CATEGORIES = [
    "Clock",
    "Teddy Bear",
    "Bowl",
    "Apple",
    "Banana",
    "Vase",
    "Mug/Cup",
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
Your task is to identify which category the object belongs to and estimate its height in centimeters (`height_cm`).

Known object categories:
{categories_list}

Supplementary information:
- Reference height: The pedestal/box the objects sit upon is 7.0 cm high (estimate only the object's own height, excluding the pedestal/box). In rare cases there may be objects that are not listed.
- Known possible objects (roughly from smallest to tallest, sometimes there are multiple candidates per category):
  1. Book laying down (~1.5 cm)
  2. Single Banana laying down (~3.5 cm)
  3. Bowl (~7 cm)
  4. Apple 1 (pinkish red) (~8 cm)
  5. Apple 2 (dark red) (~10 cm)
  6. Cake (~10 cm)
  7. Mug/Cup (~11 cm)
  8. Alarm clock (~13 cm)
  9. Bundle of Bananas laying down (~15 cm)
  10. Smaller Vase (~17.5 cm)
  11. Book standing up (~22 cm)
  12. Bottle 500ml (~23 cm)
  13. Teddy Bear (sitting vs. standing height varies; typically taller than a 500ml bottle but smaller than the big vase, ~23 cm)
  14. Bigger Vase (~23 cm)
  15. Clock (big) (~29cm)
  16. Bottle 1.5L (~30-35 cm)

Identify the object category (using one of the known object categories above), primary color, and estimated height in cm (`height_cm`). Always provide a numeric height estimate in centimeters in `height_cm` based on the photo and the reference list above.
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
- These reference heights are more accurate and trustworthy thatn the GPU server estimates. In rare cases there may be objects that are not listed. 
- Known possible objects (roughly from smallest to tallest, sometimes there are multiple candidates per category):
  1. Book laying down (~1.5 cm)
  2. Single Banana laying down (~3.5 cm)
  3. Bowl (~7 cm)
  4. Apple 1 (pinkish red) (~8 cm)
  5. Apple 2 (dark red) (~10 cm)
  6. Cake (~10 cm)
  7. Mug/Cup (~11 cm)
  8. Alarm clock (~13 cm)
  9. Bundle of Bananas laying down (~15 cm)
  10. Smaller Vase (~17.5 cm)
  11. Book standing up (~22 cm)
  12. Bottle 500ml (~23 cm)
  13. Teddy Bear (sitting vs. standing height varies; typically taller than a 500ml bottle but smaller than the big vase, ~23 cm)
  14. Bigger Vase (~23 cm)
  15. Clock (big) (~29cm)
  16. Bottle 1.5L (~30-35 cm)

### PRELIMINARY GPU SERVER STEREO & YOLO ESTIMATES (Note: The preliminary estimates are very inacurate and YOLO often detects the incorrect object, e.g. a TV (which is not a possible object) in the background. Trust your own reasoning over the estimates from YOLO (the GPU Server)):
{preliminary_estimates}

### INSTRUCTIONS:
1. Read 'photo_task_laptop' carefully and record the verbatim task instruction in `task_description`.
2. Cross-reference the GPU server estimates (class, bounding box, stereo height) with the visual photos for each location.
   If a preliminary GPU estimate appears noisy, misclassified, or missing, refine it using visual evidence from the photo.
3. Build an inventory listing each inspected point with its final object class, color, and height.
4. Provide the answer in `answer` as a complete, informative sentence — NEVER reply with only a bare number, single word, or bare point letter.
   Always state the direct answer AND explicitly name the relevant objects, their color or height and their point locations in a full sentence.
   - Example (counting/comparison): If the prompt is "How many objects are taller than the object at point D?", do NOT just answer "3"; instead answer:
     "3. The Vase at point B, the Clock at point C and the Book at point I are taller than the Cake at point D."
   - Example (color count): "2. The Bottle at point B and the Cup at point H have the same blue color as the Bowl at point D."
   - Example (superlative/location): "Point D. The Bottle at point D is the tallest object in the venue with an estimated height of 31.5 cm."
"""

