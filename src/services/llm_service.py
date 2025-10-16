"""
LLM service providing centralized async LLM operations.
Implements timeout, rate limiting, retry, and cost tracking for production use.
"""

import time
import asyncio
from typing import Type
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMTimeoutError(Exception):
    """Raised when LLM call exceeds timeout threshold."""


class LLMError(Exception):
    """Base exception for LLM-related errors."""


def create_llm(
    model_name: str, api_key: str, temperature: float = 0
) -> BaseChatModel:
    """
    Factory function to create chat model instances.

    Args:
        model_name: Model identifier (e.g., "gpt-4o", "gemini-2.5-flash")
        api_key: API key for the provider
        temperature: Sampling temperature (0 for deterministic)

    Returns:
        Configured chat model instance

    Raises:
        ValueError: If model provider is not supported
    """
    if "gemini" in model_name:
        return ChatGoogleGenerativeAI(
            google_api_key=api_key, model=model_name, temperature=temperature
        )
    elif "gpt" in model_name:
        return ChatOpenAI(
            api_key=api_key, model=model_name, temperature=temperature
        )
    else:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            "Model name must contain 'gpt' or 'gemini'"
        )


class LLMService:
    """
    Centralized async service for all LLM operations with production features.
    Provides timeout, rate limiting, retry logic, and cost tracking.
    """

    def __init__(
        self,
        model: BaseChatModel,
        max_retries: int = 3,
        timeout: int = 30,
        rate_limit: int = 3,
    ):
        """
        Initialize async LLM service.

        Args:
            model: Configured chat model instance
            max_retries: Maximum retry attempts for failed calls
            timeout: Timeout in seconds for each LLM call
            rate_limit: Maximum concurrent LLM requests (Semaphore)
        """
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(rate_limit)

    async def invoke_with_retry(
        self,
        messages: list[BaseMessage] | str,
        timeout: int | None = None,
    ) -> BaseMessage:
        """
        Invokes LLM with async retry, timeout, and rate limiting.

        Args:
            messages: Input messages or single prompt string
            timeout: Override default timeout (seconds)

        Returns:
            LLM response as BaseMessage

        Raises:
            LLMTimeoutError: If call exceeds timeout
            LLMError: If call fails after all retries
        """
        timeout = timeout or self.timeout
        start_time = time.time()

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((LLMError, LLMTimeoutError)),
            reraise=True,
        ):
            with attempt:
                try:
                    logger.info(
                        "llm_call_started",
                        attempt=attempt.retry_state.attempt_number,
                        timeout=timeout,
                        model=getattr(self.model, "model_name", "unknown"),
                    )

                    async with self.semaphore:
                        response = await asyncio.wait_for(
                            self.model.ainvoke(messages), timeout=timeout
                        )

                    elapsed = time.time() - start_time
                    self._log_usage(response, elapsed)
                    return response

                except asyncio.TimeoutError as e:
                    elapsed = time.time() - start_time
                    logger.error(
                        "llm_call_timeout",
                        exc_info=True,
                        elapsed=elapsed,
                        timeout=timeout,
                        attempt=attempt.retry_state.attempt_number,
                    )
                    raise LLMTimeoutError(
                        f"LLM call exceeded timeout of {timeout}s"
                    ) from e
                except Exception as e:
                    elapsed = time.time() - start_time
                    logger.error(
                        "llm_call_failed",
                        exc_info=True,
                        elapsed=elapsed,
                        attempt=attempt.retry_state.attempt_number,
                        error=str(e),
                    )
                    raise LLMError(f"LLM invocation failed: {e}") from e

    async def invoke_with_structured_output(
        self,
        messages: list[BaseMessage] | str,
        output_schema: Type[BaseModel],
        timeout: int | None = None,
    ) -> BaseModel:
        """
        Invokes LLM with structured output (function calling) using async.

        Args:
            messages: Input messages or prompt
            output_schema: Pydantic model defining expected output structure
            timeout: Override default timeout (seconds)

        Returns:
            Parsed structured output matching the schema

        Raises:
            LLMTimeoutError: If call exceeds timeout
            LLMError: If call fails or schema validation fails after all retries
        """
        timeout = timeout or self.timeout
        start_time = time.time()

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((LLMError, LLMTimeoutError)),
            reraise=True,
        ):
            with attempt:
                try:
                    logger.info(
                        "llm_structured_call_started",
                        schema=output_schema.__name__,
                        timeout=timeout,
                        attempt=attempt.retry_state.attempt_number,
                    )

                    async with self.semaphore:
                        response = await asyncio.wait_for(
                            self.model.ainvoke(messages, tools=[output_schema]),
                            timeout=timeout,
                        )

                    elapsed = time.time() - start_time

                    if not response.tool_calls:
                        raise LLMError(
                            f"Model did not return structured output "
                            f"for schema {output_schema.__name__}"
                        )

                    tool_call = response.tool_calls[0]
                    self._log_usage(response, elapsed)

                    return tool_call

                except asyncio.TimeoutError as e:
                    elapsed = time.time() - start_time
                    logger.error(
                        "llm_structured_call_timeout",
                        exc_info=True,
                        elapsed=elapsed,
                        timeout=timeout,
                        attempt=attempt.retry_state.attempt_number,
                        schema=output_schema.__name__,
                    )
                    raise LLMTimeoutError(
                        f"Structured output call exceeded timeout of {timeout}s"
                    ) from e
                except Exception as e:
                    elapsed = time.time() - start_time
                    logger.error(
                        "llm_structured_call_failed",
                        exc_info=True,
                        elapsed=elapsed,
                        attempt=attempt.retry_state.attempt_number,
                        schema=output_schema.__name__,
                        error=str(e),
                    )
                    raise LLMError(f"Structured output invocation failed: {e}") from e

    def _log_usage(self, response: BaseMessage, elapsed: float) -> None:
        """
        Logs token usage and cost information.

        Args:
            response: LLM response message
            elapsed: Elapsed time in seconds
        """
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            logger.info(
                "llm_usage",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                elapsed=elapsed,
            )
        else:
            logger.info("llm_call_completed", elapsed=elapsed)
