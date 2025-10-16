"""
Medicine service handling intent classification and medicine validation.
Coordinates router logic and validates medicines against known database.
"""

import re
from src.models.domain import AgentState
from src.models.schemas import UserIntent
from src.services.llm_service import LLMService
from src.utils.prompts import load_prompts
from src.utils.logger import get_logger

logger = get_logger(__name__)
PROMPTS = load_prompts()


class MedicineService:
    """
    Service for medicine-related operations including intent classification
    and medicine name validation.
    """

    def __init__(self, llm_service: LLMService, known_medicines: list[str]):
        """
        Initialize medicine service.

        Args:
            llm_service: LLM service for intent classification
            known_medicines: List of medicine names available in database
        """
        self.llm_service = llm_service
        self.known_medicines = [med.lower() for med in known_medicines]

    async def classify_intent_and_validate(self, state: AgentState) -> dict:
        """
        Classifies user intent and validates medicine name against known list (async).

        Args:
            state: Current agent state

        Returns:
            Dictionary with updated intent and current_medicines

        Raises:
            Exception: LLM errors are propagated (fail-fast)
        """
        logger.info("intent_classification_started")
        last_message = state["messages"][-1].content

        prompt_template = PROMPTS["intent_classification"]["prompt_template"]
        prompt = prompt_template.format(user_message=last_message)

        try:
            tool_call = await self.llm_service.invoke_with_structured_output(
                prompt, UserIntent
            )

            intent = tool_call["args"]["intent"]
            medicine = tool_call["args"].get("medicine_name")

            logger.info(
                "intent_classification_completed",
                intent=intent,
                medicine=medicine,
            )

            current_medicines = state.get("current_medicines", [])

            if medicine and intent == "pregunta_medicamento":
                intent = self._validate_and_update_medicines(
                    medicine, current_medicines
                )

            return {"intent": intent, "current_medicines": current_medicines}

        except Exception as e:
            logger.error(
                "intent_classification_failed",
                exc_info=True,
                error=str(e),
                default_intent="pregunta_general",
            )
            return {"intent": "pregunta_general"}

    def _validate_and_update_medicines(
        self, medicine: str, current_medicines: list[str]
    ) -> str:
        """
        Validates medicine against known list and updates current medicines.

        Args:
            medicine: Extracted medicine name
            current_medicines: List of currently tracked medicines

        Returns:
            Intent string ("pregunta_medicamento" or "pregunta_no_autorizada")
        """
        medicine_lower = medicine.lower()

        matched_medicine = self._find_matching_medicine(medicine_lower)

        if matched_medicine:
            logger.info("medicine_validated", medicine=matched_medicine)
            if matched_medicine not in current_medicines:
                current_medicines.append(matched_medicine)
            return "pregunta_medicamento"
        else:
            logger.warning(
                "medicine_not_in_database",
                medicine=medicine_lower,
                action="routing_to_unauthorized",
            )
            return "pregunta_no_autorizada"

    def _find_matching_medicine(self, medicine: str) -> str | None:
        """
        Finds matching medicine using word boundary regex to avoid partial matches.

        Args:
            medicine: Medicine name to match

        Returns:
            Matched medicine name or None if not found
        """
        for known_med in self.known_medicines:
            if re.search(r"\b" + re.escape(known_med) + r"\b", medicine):
                return known_med
        return None

    def get_unauthorized_medicine_message(self) -> str:
        """
        Generates message for unauthorized medicine queries.

        Returns:
            Formatted message listing available medicines
        """
        known_medicines_str = ", ".join(med.title() for med in self.known_medicines)
        template = PROMPTS["conversation_responses"]["unauthorized_medicine_template"]
        return template.format(known_medicines=known_medicines_str)

    def get_retrieval_failure_message(self, medicine_name: str) -> str:
        """
        Generates message for retrieval failures.

        Args:
            medicine_name: Name of the medicine that was searched

        Returns:
            Formatted failure message
        """
        template = PROMPTS["conversation_responses"]["retrieval_failure_template"]
        return template.format(medicine_name=medicine_name.title())
