"""
LLM structured output schemas for function calling and structured extraction.
All models use Field() with descriptions for clarity and LLM context.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


class UserIntent(BaseModel):
    """
    Classifies user's intent for conversation routing.
    Used by the router LLM to determine the conversation path.
    """

    intent: Literal[
        "pregunta_medicamento", "pregunta_general", "saludo_despedida"
    ] = Field(description="Primary intent category of the user's message")

    medicine_name: Optional[str] = Field(
        default=None,
        description=(
            "Medicine name extracted from the question. "
            "Only populate if intent is 'pregunta_medicamento'"
        ),
        max_length=100,
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "intent": "pregunta_medicamento",
                "medicine_name": "ibuprofeno",
            }
        }
    )


class MedicineToolInput(BaseModel):
    """
    Input schema for the medicine information retrieval tool.
    Defines the query parameter for RAG search.
    """

    query: str = Field(
        description=(
            "Specific question about the medicine to search for in the database. "
            "Should be clear and focused on a single aspect (e.g., dosage, "
            "side effects, contraindications)"
        ),
        min_length=3,
        max_length=500,
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "¿Cuáles son los efectos secundarios del ibuprofeno?",
            }
        }
    )
