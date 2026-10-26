from pydantic import BaseModel, Field
from typing import List, Optional

class Waypoint(BaseModel):
    name: str = Field(description="Name of the Waypoint.")

class Waypoints(BaseModel):
    waypoints: List[Waypoint]