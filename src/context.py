from __future__ import annotations

from config import logger
from llm import LLMClient, LLMResponse
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

_COMPACTION_PROMPT = (
    "Summarize the conversation so far into a concise system message that preserves all "
    "important facts, decisions, tool results, and user preferences. "
    "Omit pleasantries and redundant exchanges. Be dense and complete."
)


class ContextManager:
    def __init__(self, context_window: int, compaction_threshold: float) -> None:
        self._window = context_window
        self._compact_at = compaction_threshold
        self._used: int = 0

    def update(self, response: LLMResponse) -> None:
        self._used = response.input_tokens

    def should_compact(self) -> bool:
        return self._used / self._window >= self._compact_at

    async def compact(
        self,
        history: list[ChatCompletionMessageParam],
        llm: LLMClient,
    ) -> list[ChatCompletionMessageParam]:
        """Summarise the oldest half of history into a single system message."""
        if len(history) < 4:
            return history

        mid = len(history) // 2
        to_summarise = history[:mid]
        keep = history[mid:]

        summary_messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(role="system", content=_COMPACTION_PROMPT),
            *to_summarise,
        ]
        try:
            response = await llm.chat(summary_messages)
            summary = response.content or "(no summary)"
        except Exception as exc:
            logger.warning(f"compaction LLM call failed: {exc}")
            summary = "(compaction failed — prior context truncated)"

        compacted_system = ChatCompletionUserMessageParam(role="user", content=f"[Compacted context summary]\n{summary}")
        result = [compacted_system, *keep]
        logger.info(f"compacted {len(to_summarise)} messages → 1 summary + {len(keep)} kept")
        self._used = 0  # reset; will be updated after next LLM call
        return result
