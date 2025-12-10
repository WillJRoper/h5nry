"""Google Gemini LLM client implementation."""

from __future__ import annotations

from typing import Any

import anyio
import google.generativeai as genai

from h5nry.llm.base import LLMClient, LLMResponse, Message


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

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,  # noqa: ARG002
    ) -> LLMResponse:
        """Send a chat request to Gemini.

        Args:
            messages: Conversation history
            tools: Optional list of tool definitions (not implemented)

        Returns:
            LLM response

        Note:
            Gemini's tool/function calling support may be limited compared to
            OpenAI and Anthropic. This is a basic implementation.
            Tool calling is not currently implemented.
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

        # Convert messages to chat history format
        history = gemini_messages[:-1] if len(gemini_messages) > 1 else []
        latest_message = gemini_messages[-1]["parts"][0] if gemini_messages else ""

        # Start chat session (sync API, wrap in thread)
        def _sync_chat():
            chat = self.model_obj.start_chat(history=history)
            response = chat.send_message(
                latest_message,
                generation_config=generation_config,
            )
            return response

        response = await anyio.to_thread.run_sync(_sync_chat)

        # Parse response
        content = response.text if hasattr(response, "text") else None

        # Note: Gemini function calling works differently and may require
        # additional implementation. For now, we return empty tool_calls.
        # TODO: Implement Gemini function calling properly if needed
        tool_calls = []

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=None,
        )
