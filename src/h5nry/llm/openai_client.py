"""OpenAI LLM client implementation."""

from __future__ import annotations

import json
from typing import Any

import anyio
from openai import AsyncOpenAI

from h5nry.llm.base import LLMClient, LLMResponse, Message, ToolCall


class OpenAIClient(LLMClient):
    """OpenAI LLM client."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ):
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        super().__init__(model, temperature, max_tokens)
        self.client = AsyncOpenAI(api_key=api_key)

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert our message format to OpenAI format.

        Args:
            messages: List of messages in our format

        Returns:
            List of messages in OpenAI format
        """
        openai_messages = []
        for msg in messages:
            openai_msg: dict[str, Any] = {
                "role": msg["role"],
                "content": msg["content"],
            }
            if "name" in msg and msg["name"]:
                openai_msg["name"] = msg["name"]
            if "tool_call_id" in msg and msg["tool_call_id"]:
                openai_msg["tool_call_id"] = msg["tool_call_id"]
            if "tool_calls" in msg and msg["tool_calls"]:
                openai_msg["tool_calls"] = msg["tool_calls"]
            openai_messages.append(openai_msg)
        return openai_messages

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Send a chat request to OpenAI.

        Args:
            messages: Conversation history
            tools: Optional list of tool definitions

        Returns:
            LLM response
        """
        openai_messages = self._convert_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": self.temperature,
        }

        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        # Make API call
        response = await self.client.chat.completions.create(**kwargs)

        # Parse response
        message = response.choices[0].message
        content = message.content

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=response.choices[0].finish_reason,
        )
