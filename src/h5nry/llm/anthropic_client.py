"""Anthropic LLM client implementation."""

from __future__ import annotations

from typing import Any

from anthropic import AsyncAnthropic

from h5nry.llm.base import LLMClient, LLMResponse, Message, ToolCall


class AnthropicClient(LLMClient):
    """Anthropic LLM client."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ):
        """Initialize Anthropic client.

        Args:
            api_key: Anthropic API key
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        super().__init__(model, temperature, max_tokens)
        self.client = AsyncAnthropic(api_key=api_key)
        # Anthropic requires max_tokens
        if not self.max_tokens:
            self.max_tokens = 4096

    def _convert_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert our message format to Anthropic format.

        Anthropic separates system messages from the conversation.

        Args:
            messages: List of messages in our format

        Returns:
            Tuple of (system_prompt, anthropic_messages)
        """
        system_prompt = None
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                anthropic_msg: dict[str, Any] = {
                    "role": msg["role"] if msg["role"] != "tool" else "user",
                    "content": msg["content"],
                }
                anthropic_messages.append(anthropic_msg)

        return system_prompt, anthropic_messages

    def _convert_tools(
        self, tools: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]]:
        """Convert OpenAI-style tools to Anthropic format.

        Args:
            tools: Tools in OpenAI format

        Returns:
            Tools in Anthropic format
        """
        if not tools:
            return []

        anthropic_tools = []
        for tool in tools:
            if tool["type"] == "function":
                func = tool["function"]
                anthropic_tools.append(
                    {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "input_schema": func["parameters"],
                    }
                )

        return anthropic_tools

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Send a chat request to Anthropic.

        Args:
            messages: Conversation history
            tools: Optional list of tool definitions

        Returns:
            LLM response
        """
        system_prompt, anthropic_messages = self._convert_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        if tools:
            anthropic_tools = self._convert_tools(tools)
            if anthropic_tools:
                kwargs["tools"] = anthropic_tools

        # Make API call
        response = await self.client.messages.create(**kwargs)

        # Parse response
        content = None
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=response.stop_reason,
        )
