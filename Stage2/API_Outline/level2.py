from google import genai
from pydantic import BaseModel, Field
from typing import List, Optional
from category import Categorys, Category
from Stage2.API_Outline.waypoint import Waypoints, Waypoint

client = genai.Client()

promptTask = """
Analyze the attached photo of the laptop screen displaying task instructions.
The task instruction is in the format of:
Please find the point where X is located.
There are two points we need to find.
X stands for an object category. 

Known object categories: Clock, Teddy Bear, Bowl, Apple, Banana, Vase, Cup, Cake, Bottle, Book

Return the categories we need to visit. 
"""

interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input=prompt,
    response_format={
        "type": "text",
        "mime_type": "application/json",
        "schema": Waypoints.model_json_schema()
    },
)

recipe = Waypoints.model_validate_json(interaction.output_text)
print(recipe)

promptTask = """
Analyze the attached photo. You task is to identify the category the object in the photo belongs to.

Known object categories: Clock, Teddy Bear, Bowl, Apple, Banana, Vase, Cup, Cake, Bottle, Book

Return the category of the object in the picture.
"""
