from pydantic import BaseModel, Field
from typing import List, Optional

class Category(BaseModel):
    category: str

class Categories(BaseModel):
    categories: List[Category]