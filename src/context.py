from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path

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

_TOOL_RESULTS_ARCHIVE = Path("./data/archive/tool_results")


class ContextManager:
    def __init__(self, context_window: int, keep_recent: int = 10,
                 keep_ratio: float = 0.25) -> None:
        self._window = context_window
        self._keep_recent = keep_recent
        self._keep_ratio = keep_ratio
        self._used: int = 0

    def update(self, response: LLMResponse) -> None:
        self._used = response.input_tokens

    def should_compact(self) -> bool:
        return self._used > self._window

    # ── tier 1: microcompact ──

    def microcompact(
        self, history: list[ChatCompletionMessageParam]
    ) -> list[ChatCompletionMessageParam]:
        """Truncate old tool results to save tokens. No LLM call needed."""
        if len(history) < self._keep_recent + 2:
            return history

        boundary = len(history) - self._keep_recent
        archived: list[dict] = []
        result: list[ChatCompletionMessageParam] = []

        for i, msg in enumerate(history):
            if i < boundary and msg.get("role") == "tool":
                content = str(msg.get("content", ""))
                if len(content) > 200:
                    archived.append({
                        "tool_call_id": msg.get("tool_call_id", ""),
                        "content": content,
                        "archived_at": datetime.now(timezone.utc).isoformat(),
                    })
                    truncated = content[:100] + "..."
                    result.append({**msg, "content": truncated})  # type: ignore[arg-type]
                    continue
            result.append(msg)

        if archived:
            self._archive_tool_results(archived)
            logger.info(f"microcompact: truncated {len(archived)} tool results")

        return result

    def _archive_tool_results(self, entries: list[dict]) -> None:
        _TOOL_RESULTS_ARCHIVE.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        path = _TOOL_RESULTS_ARCHIVE / f"{ts}.jsonl"
        with open(path, "a") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    # ── compaction ──

    async def compact(
        self,
        history: list[ChatCompletionMessageParam],
        llm: LLMClient,
        memory_flush_fn: Callable[[list[ChatCompletionMessageParam]], Awaitable] | None = None,
    ) -> list[ChatCompletionMessageParam]:
        """Compaction: pre-flush → microcompact the half being summarized → full LLM summary."""
        if len(history) < 4:
            return history

        # tier 1: pre-compaction memory flush
        if memory_flush_fn is not None:
            try:
                await memory_flush_fn(history)
            except Exception as exc:
                logger.error(f"pre-compaction memory flush failed: {exc}")

        # tier 2: split history, microcompact the portion being summarized
        keep_count = max(2, int(len(history) * self._keep_ratio))
        split = len(history) - keep_count
        to_summarise = self.microcompact(history[:split])
        keep = history[split:]

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

        compacted_system = ChatCompletionUserMessageParam(
            role="user", content=f"[Compacted context summary]\n{summary}"
        )
        result = [compacted_system, *keep]
        logger.info(f"compacted {len(to_summarise)} messages → 1 summary + {len(keep)} kept")
        self._used = 0
        return result
