"""
Unit tests for MedicineService.
Tests intent classification, medicine validation, and matching logic.
"""

import pytest
from unittest.mock import Mock
from langchain_core.messages import HumanMessage

from src.services.medicine_service import MedicineService
from src.services.llm_service import LLMService
from src.models.domain import AgentState


@pytest.fixture
def llm_service():
    """Mock LLM service for testing."""
    return Mock(spec=LLMService)


@pytest.fixture
def medicine_service(llm_service, known_medicines):
    """Create MedicineService with mocked dependencies."""
    return MedicineService(llm_service, known_medicines)


class TestMedicineMatching:
    """Tests for medicine name matching logic."""

    def test_exact_match(self, medicine_service):
        """Should match exact medicine name."""
        # Arrange
        medicine = "ibuprofeno"
        
        # Act
        result = medicine_service._find_matching_medicine(medicine)
        
        # Assert
        assert result == "ibuprofeno"

    def test_case_insensitive_match(self, medicine_service):
        """Should match regardless of case."""
        # Arrange
        medicine = "IBUPROFENO"
        
        # Act
        result = medicine_service._find_matching_medicine(medicine.lower())
        
        # Assert
        assert result == "ibuprofeno"

    def test_partial_match_with_word_boundaries(self, medicine_service):
        """Should match medicine name within longer string."""
        # Arrange
        medicine = "ibuprofeno 600 mg"
        
        # Act
        result = medicine_service._find_matching_medicine(medicine)
        
        # Assert
        assert result == "ibuprofeno"

    def test_no_partial_substring_match(self, medicine_service):
        """Should NOT match if medicine is substring without word boundary."""
        # Arrange
        medicine = "ibuprofenoide"  # Contains "ibuprofeno" but not as word
        
        # Act
        result = medicine_service._find_matching_medicine(medicine)
        
        # Assert
        assert result is None

    def test_no_match_unknown_medicine(self, medicine_service):
        """Should return None for unknown medicine."""
        # Arrange
        medicine = "aspirina"  # Not in known_medicines
        
        # Act
        result = medicine_service._find_matching_medicine(medicine)
        
        # Assert
        assert result is None

    def test_match_with_brand_name(self, medicine_service):
        """Should match medicine by brand name if in known list."""
        # Arrange - "nolotil" is in the known_medicines fixture
        medicine = "nolotil 575"
        
        # Act
        result = medicine_service._find_matching_medicine(medicine)
        
        # Assert
        assert result == "nolotil"


class TestIntentValidation:
    """Tests for medicine validation and intent determination."""

    def test_validate_known_medicine(self, medicine_service):
        """Should accept known medicine and update state."""
        # Arrange
        medicine = "ibuprofeno"
        current_medicines = []
        
        # Act
        intent = medicine_service._validate_and_update_medicines(
            medicine, current_medicines
        )
        
        # Assert
        assert intent == "pregunta_medicamento"
        assert "ibuprofeno" in current_medicines

    def test_validate_unknown_medicine(self, medicine_service):
        """Should reject unknown medicine."""
        # Arrange
        medicine = "aspirina"
        current_medicines = []
        
        # Act
        intent = medicine_service._validate_and_update_medicines(
            medicine, current_medicines
        )
        
        # Assert
        assert intent == "pregunta_no_autorizada"
        assert len(current_medicines) == 0

    def test_no_duplicate_medicines(self, medicine_service):
        """Should not add duplicate medicine to state."""
        # Arrange
        medicine = "ibuprofeno"
        current_medicines = ["ibuprofeno"]
        
        # Act
        medicine_service._validate_and_update_medicines(
            medicine, current_medicines
        )
        
        # Assert
        assert current_medicines == ["ibuprofeno"]
        assert len(current_medicines) == 1


class TestIntentClassification:
    """Tests for full intent classification flow."""

    @pytest.mark.asyncio
    async def test_classify_medicine_question(self, medicine_service, llm_service):
        """Should classify medicine question correctly."""
        # Arrange
        state: AgentState = {
            "messages": [HumanMessage(content="¿Qué es el ibuprofeno?")],
            "intent": None,
            "current_medicines": [],
            "summary": "",
            "turn_count": 0,
        }

        llm_service.invoke_with_structured_output.return_value = {
            "args": {
                "intent": "pregunta_medicamento",
                "medicine_name": "ibuprofeno",
            }
        }

        # Act
        result = await medicine_service.classify_intent_and_validate(state)

        # Assert
        assert result["intent"] == "pregunta_medicamento"
        assert "ibuprofeno" in result["current_medicines"]

    @pytest.mark.asyncio
    async def test_classify_unknown_medicine_question(self, medicine_service, llm_service):
        """Should route unknown medicine to unauthorized."""
        # Arrange
        state: AgentState = {
            "messages": [HumanMessage(content="¿Qué es la aspirina?")],
            "intent": None,
            "current_medicines": [],
            "summary": "",
            "turn_count": 0,
        }

        llm_service.invoke_with_structured_output.return_value = {
            "args": {
                "intent": "pregunta_medicamento",
                "medicine_name": "aspirina",
            }
        }

        # Act
        result = await medicine_service.classify_intent_and_validate(state)

        # Assert
        assert result["intent"] == "pregunta_no_autorizada"
        assert len(result["current_medicines"]) == 0

    @pytest.mark.asyncio
    async def test_classify_general_question(self, medicine_service, llm_service):
        """Should classify general question correctly."""
        # Arrange
        state: AgentState = {
            "messages": [HumanMessage(content="Hola")],
            "intent": None,
            "current_medicines": [],
            "summary": "",
            "turn_count": 0,
        }

        llm_service.invoke_with_structured_output.return_value = {
            "args": {
                "intent": "saludo_despedida",
                "medicine_name": None,
            }
        }

        # Act
        result = await medicine_service.classify_intent_and_validate(state)

        # Assert
        assert result["intent"] == "saludo_despedida"

    @pytest.mark.asyncio
    async def test_fallback_on_llm_error(self, medicine_service, llm_service):
        """Should fallback to general question on LLM error."""
        # Arrange
        state: AgentState = {
            "messages": [HumanMessage(content="Test")],
            "intent": None,
            "current_medicines": [],
            "summary": "",
            "turn_count": 0,
        }

        llm_service.invoke_with_structured_output.side_effect = Exception(
            "LLM error"
        )

        # Act
        result = await medicine_service.classify_intent_and_validate(state)

        # Assert
        assert result["intent"] == "pregunta_general"


class TestMessageGeneration:
    """Tests for message generation methods."""

    def test_unauthorized_medicine_message(self, medicine_service):
        """Should generate message listing known medicines."""
        # Act
        message = medicine_service.get_unauthorized_medicine_message()
        
        # Assert
        assert "Ibuprofeno" in message  # Title case
        assert "Paracetamol" in message
        assert "no tengo información" in message.lower()

    def test_retrieval_failure_message(self, medicine_service):
        """Should generate failure message with medicine name."""
        # Arrange
        medicine_name = "ibuprofeno"
        
        # Act
        message = medicine_service.get_retrieval_failure_message(medicine_name)
        
        # Assert
        assert "Ibuprofeno" in message  # Title case
        assert "no he podido encontrar" in message.lower()

