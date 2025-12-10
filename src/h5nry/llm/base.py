"""Base classes for LLM client abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, TypedDict

from pydantic import BaseModel


class Message(TypedDict, total=False):
    """Message in a conversation."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None  # For tool messages
    tool_call_id: str | None  # For tool responses
    tool_calls: list[dict[str, Any]] | None  # For assistant tool calls


class ToolCall(BaseModel):
    """Represents a tool call request from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


class LLMResponse(BaseModel):
    """Response from an LLM."""

    content: str | None
    tool_calls: list[ToolCall]
    finish_reason: str | None = None


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(
        self, model: str, temperature: float = 0.1, max_tokens: int | None = None
    ):
        """Initialize LLM client.

        Args:
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Send a chat request to the LLM.

        Args:
            messages: Conversation history
            tools: Optional list of tool definitions

        Returns:
            LLM response with content and/or tool calls
        """
        pass
