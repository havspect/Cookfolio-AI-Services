# %%
from typing import List
from pydantic import BaseModel

class RecipeIngredient(BaseModel):
    """Represents a single ingredient in a recipe.
    """
    name: str
    quantity: str | None
    unit: str | None


class RecipeInstruction(BaseModel):
    """Represents a single step in the recipe instructions.
    """
    step: int
    description: str

class RecipeInstructionList(BaseModel):
    """Represents a list of recipe instructions.
    """
    instructions: List[RecipeInstruction]

# %%
class Recipe(BaseModel):
    title: str
    description: str
    ingredients: List[RecipeIngredient]
    instructions: List[RecipeInstruction]  