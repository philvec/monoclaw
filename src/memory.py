from __future__ import annotations

import asyncio
from config import logger, ToolsConfig, SYSERR
from datetime import datetime, timezone
from pathlib import Path

from llm import LLMClient
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam
from pydantic import BaseModel

_BASE_SYSTEM_PROMPT = """\
You are monoclaw, a minimal personal AI assistant.
You have a single continuous session shared across all channels — your conversation history is fully persistent and restored across restarts. \
Do not claim to lack memory; if prior context is visible in the conversation, use it. \
After each turn, key facts are automatically extracted and saved to long-term memory (you will see [MEMORY SAVED] notes in the conversation when this happens).
You may have access to tools for file operations, shell execution, web search, and web fetch.
All file and shell operations run inside the workspace directory.
Be concise. Do not pad responses. When using tools, prefer the simplest approach.
"""

_EXTRACT_MEMORIES_PROMPT = """\
Review the conversation below and extract any facts worth remembering for future sessions.
Include user preferences, personality traits, long-term goals, task-specific decisions, and pending work.
Ignore pleasantries and one-off queries that won't matter later.
Return an empty list if there is nothing worth remembering.

Conversation:
"""

_MEMORY_BASE_PATH = Path("./data/memory")
_MEMORY_MD_HEADER = "# monoclaw memory\n\n"
_MEMORY_FILE = _MEMORY_BASE_PATH / "MEMORY.md"
_MEMORY_LOCK = asyncio.Lock()


class MemoryExtractionResponse(BaseModel):
    entries: list[str]


class MemoryManager:
    def __init__(self, llm: LLMClient, cfg: ToolsConfig) -> None:
        self._llm = llm
        self._cfg = cfg
        _MEMORY_BASE_PATH.mkdir(parents=True, exist_ok=True)

    def build_system_prompt(self) -> str:
        """Concatenate memory → base instructions."""
        parts: list[str] = []

        if _MEMORY_FILE.exists():
            try:
                parts.append(_MEMORY_FILE.read_text().strip())
            except Exception as exc:
                logger.error(err := f"failed to read memory: {exc}")
                parts.append(f"[{SYSERR} — memory is missing ({err})]")

        parts.append(_BASE_SYSTEM_PROMPT)
        return "\n\n---\n\n".join(p for p in parts if p)

    async def extract_memories(self, conversation: list[ChatCompletionMessageParam]) -> list[str]:
        """Post-turn: ask LLM to extract memorable facts and append to memory file. Returns saved entries."""
        if not conversation:
            return []

        convo_text = self._format_conversation(conversation)
        prompt_messages: list[ChatCompletionMessageParam] = [
            ChatCompletionUserMessageParam(role="user", content=_EXTRACT_MEMORIES_PROMPT + convo_text)
        ]
        response = await self._llm.chat(prompt_messages, response_model=MemoryExtractionResponse)
        if not isinstance(response.parsed, MemoryExtractionResponse):
            return []
        parsed = response.parsed

        if parsed.entries:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            block = "\n".join(f"[{ts}] {e}" for e in parsed.entries)
            async with _MEMORY_LOCK:
                existing = _MEMORY_FILE.read_text() if _MEMORY_FILE.exists() else _MEMORY_MD_HEADER
                _MEMORY_FILE.write_text(existing.rstrip() + "\n" + block + "\n")
            logger.info(f"saved {len(parsed.entries)} memory entries")

        return parsed.entries

    def _format_conversation(self, messages: list[ChatCompletionMessageParam]) -> str:
        lines = []
        for msg in messages[-self._cfg.memory_ctx_trunc_n :]:
            if content := msg.get("content", ""):
                role = msg.get("role", "").upper()
                lines.append(f"{role}: {str(content)[:self._cfg.memory_msg_max_len]}")
        return "\n".join(lines)
