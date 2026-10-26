from google import genai
from pydantic import BaseModel, Field
from typing import List, Optional
from category import Categorys, Category
from waypoint import Waypoints, Waypoint

client = genai.Client()

promptTask = """
You are the perception and reasoning module of a TurtleBot4 Lite robot equipped with a camera.

### CONTEXT & INPUTS:
The robot has already visited 10 fixed locations:
- Point A: Task Center (contains a laptop displaying task instructions)
- Points B through J: 9 locations with randomly placed objects

You are provided with:
1. Photo of the laptop at Point A displaying the task instructions.
2. Photos for each location: photo_point_B through photo_point_J.
3. Preliminary robot sensor estimates (object class and measured height) for Points B through J. 

Known possible objects (sorted roughly from smallest to tallest):
- Apple
- Cake
- Mug
- Alarm clock
- Banana
- Book
- Bottle (500ml)
- Teddy Bear (sitting vs. standing height varies; typically taller than a 500ml bottle)
- Vase
- Bottle (1.5L)
*(Other unexpected objects may also appear)*

---

### INSTRUCTIONS:

Proceed through the following steps systematically:

#### STEP 1: Transcribe the Tasks
Read the laptop screen at Point A carefully and transcribe the task verbatim:
- An example for a task could be: Please count how many objects in the venue have the same color as the object at Point D. 

#### STEP 2: Ground Truth Inventory (Points B to J)
Cross-reference the sensor estimates with the actual photos for Points B to J. If the initial sensor estimate appears incorrect, correct it based on visual evidence.
Create an internal inventory listing:
- Location (B through J)
- Final Object Class
- Estimated Relative/Reported Height

#### STEP 3: Final Answers
Answer the task accurately using the inventory from Step 2.

---

### PRELIMINARY SENSOR ESTIMATES:
"""
#TODO
#- Point B: [Class, Height]
#- Point C: [Class, Height]
#- Point D: [Class, Height]
#- Point E: [Class, Height]
#- Point F: [Class, Height]
#- Point G: [Class, Height]
#- Point H: [Class, Height]
#- Point I: [Class, Height]
#- Point J: [Class, Height]

interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input=prompt,
    response_format={
        "type": "text",
    },
)