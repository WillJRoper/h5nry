"""Google Gemini LLM client implementation."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

import anyio
import google.generativeai as genai

from h5nry.llm.base import LLMClient, LLMResponse, Message, ToolCall


class GeminiClient(LLMClient):
    """Google Gemini LLM client."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-pro",
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ):
        """Initialize Gemini client.

        Args:
            api_key: Google API key
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        super().__init__(model, temperature, max_tokens)
        genai.configure(api_key=api_key)
        self.model_obj = genai.GenerativeModel(model)

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert our message format to Gemini format.

        Args:
            messages: List of messages in our format

        Returns:
            List of messages in Gemini format
        """
        # Gemini uses "user" and "model" roles
        # System messages are handled separately via generation config
        gemini_messages = []

        for msg in messages:
            role = msg["role"]
            if role == "assistant":
                role = "model"
            elif role in ("system", "tool"):
                role = "user"  # Gemini doesn't have explicit system/tool roles

            gemini_messages.append(
                {
                    "role": role,
                    "parts": [msg["content"]],
                }
            )

        return gemini_messages

    def _convert_tools(
        self, tools: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]]:
        """Convert OpenAI-style tools to Gemini format.

        Args:
            tools: Tools in OpenAI format

        Returns:
            Tools in Gemini format
        """
        if not tools:
            return []

        gemini_tools = []
        for tool in tools:
            if tool["type"] == "function":
                func = tool["function"]
                gemini_tools.append(
                    {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "parameters": func["parameters"],
                    }
                )

        return gemini_tools

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Send a chat request to Gemini.

        Args:
            messages: Conversation history
            tools: Optional list of tool definitions

        Returns:
            LLM response

        Note:
            Gemini function calling is supported but behavior may differ
            slightly from OpenAI/Anthropic.
        """
        # Extract system message if present (filter it out)
        filtered_messages = []

        for msg in messages:
            if msg["role"] != "system":
                filtered_messages.append(msg)

        gemini_messages = self._convert_messages(filtered_messages)

        # Build generation config
        generation_config = {
            "temperature": self.temperature,
        }
        if self.max_tokens:
            generation_config["max_output_tokens"] = self.max_tokens

        # Convert tools if provided
        gemini_tool_config = None
        if tools:
            gemini_tool_funcs = self._convert_tools(tools)
            if gemini_tool_funcs:
                gemini_tool_config = [
                    genai.types.Tool(function_declarations=gemini_tool_funcs)
                ]

        # Convert messages to chat history format
        history = gemini_messages[:-1] if len(gemini_messages) > 1 else []
        latest_message = gemini_messages[-1]["parts"][0] if gemini_messages else ""

        # Start chat session (sync API, wrap in thread)
        def _sync_chat():
            chat = self.model_obj.start_chat(history=history)
            response = chat.send_message(
                latest_message,
                generation_config=generation_config,
                tools=gemini_tool_config if gemini_tool_config else None,
            )
            return response

        response = await anyio.to_thread.run_sync(_sync_chat)

        # Parse response
        content = None
        tool_calls = []

        # Check if response has text content
        # Sometimes Gemini raises ValueError when only function calls exist
        if hasattr(response, "text"):
            with suppress(ValueError):
                content = response.text

        # Check for function calls
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                for part in candidate.content.parts:
                    if hasattr(part, "function_call"):
                        fc = part.function_call
                        # Convert Gemini function call to our format
                        tool_calls.append(
                            ToolCall(
                                id=f"call_{fc.name}",  # Gemini doesn't provide IDs
                                name=fc.name,
                                arguments=dict(fc.args),
                            )
                        )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=None,
        )
