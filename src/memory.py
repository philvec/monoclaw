from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from config import logger, ToolsConfig, SYSERR
from llm import LLMClient
from memory_store import MemoryStore, MemoryEntry
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam
from pydantic import BaseModel, Field, field_validator

# ── system prompt ──

_BASE_SYSTEM_PROMPT = """\
You are a personal AI assistant.
You have a single continuous session shared across all channels — your conversation history \
is fully persistent and restored across restarts. \
You have have access to context AND memory; if prior context is visible in the conversation, use it. \
After each turn, key facts are automatically extracted and saved to long-term memory \
(you will see [MEMORY SAVED] notes in the conversation when this happens).

Your long-term memory index is loaded below. It lists all stored memories by slug and one-line description. \
For full details on any memory, use memory_read with its slug. \
To find relevant memories not in the index, use memory_search with keywords — it searches across all memory content. \
Before answering questions about prior context, preferences, or decisions, search your memory.

You may have access to tools for file operations, shell execution, web search, and web fetch.
All file and shell operations run inside the workspace directory.
Be concise. Do not pad responses. When using tools, prefer the simplest approach.
"""

# ── extraction prompts ──

_EXTRACT_MEMORIES_PROMPT = """\
Review the conversation and list facts worth remembering for future sessions.
If a slug from the existing index covers the same topic, reuse that slug (this updates it).

Each memory needs: slug (lowercase-hyphens), type (user/project/reference/feedback), content (the fact).

Current memory index:
{index}

Conversation:
{conversation}
"""

_FLUSH_MEMORIES_PROMPT = """\
The conversation is about to be summarized. Save any important facts that should persist.
Reuse existing slugs to update; use new slugs for new facts.

Each memory needs: slug (lowercase-hyphens), type (user/project/reference/feedback), content (the fact).

Current memory index:
{index}

Conversation:
{conversation}
"""

# ── models ──


class MemoryOperation(BaseModel):
    """Schema aligned with what llama-cpp reliably produces via grammar constraint."""

    slug: str = Field(description="Unique identifier, lowercase with hyphens (e.g. 'user-prefers-rust')")
    type: str = Field(description="Category: 'user', 'project', 'reference', or 'feedback'")
    content: str = Field(description="The memory text to store")

    @field_validator("type", mode="before")
    @classmethod
    def lowercase_type(cls, v: str) -> str:
        return v.strip().lower() if isinstance(v, str) else v


class MemoryExtractionResponse(BaseModel):
    operations: list[MemoryOperation] = Field(
        description="List of memories to save. Empty list if nothing worth remembering."
    )


# ── lock ──

_MEMORY_LOCK = asyncio.Lock()


# ── manager ──


class MemoryManager:
    def __init__(self, llm: LLMClient, cfg: ToolsConfig, store: MemoryStore) -> None:
        self._llm = llm
        self._cfg = cfg
        self._store = store

    def build_system_prompt(self) -> str:
        """Concatenate memory index → base instructions."""
        parts: list[str] = [_BASE_SYSTEM_PROMPT]
        try:
            index = self._store.generate_index_md()
            if index:
                parts.append(index.strip())
        except Exception as exc:
            logger.error(err := f"failed to generate memory index: {exc}")
            parts.append(f"[{SYSERR} — memory index unavailable ({err})]")

        return "\n\n---\n\n".join(p for p in parts if p)

    async def extract_memories(
        self, conversation: list[ChatCompletionMessageParam]
    ) -> list[MemoryOperation]:
        """Post-turn: extract and apply memory operations. Returns ops performed."""
        return await self._run_extraction(conversation, _EXTRACT_MEMORIES_PROMPT)

    async def flush_memories(
        self, conversation: list[ChatCompletionMessageParam]
    ) -> list[MemoryOperation]:
        """Pre-compaction: aggressively save context before it's summarized away."""
        return await self._run_extraction(conversation, _FLUSH_MEMORIES_PROMPT)

    async def _run_extraction(
        self,
        conversation: list[ChatCompletionMessageParam],
        prompt_template: str,
    ) -> list[MemoryOperation]:
        if not conversation:
            return []

        convo_text = self._format_conversation(conversation)
        index = self._store.generate_index_md() or "(no memories stored yet)"
        prompt = prompt_template.format(index=index, conversation=convo_text)

        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionUserMessageParam(role="user", content=prompt)
        ]
        response = await self._llm.chat(messages, response_model=MemoryExtractionResponse)
        if not isinstance(response.parsed, MemoryExtractionResponse):
            raise ValueError(
                f"extraction failed: LLM did not produce valid MemoryExtractionResponse "
                f"(finish_reason={response.finish_reason}, content={response.content[:200]!r})"
            )

        ops = response.parsed.operations
        applied: list[tuple[str, MemoryOperation]] = []  # (action, op)

        async with _MEMORY_LOCK:
            for op in ops:
                try:
                    action = await self._apply_operation(op)
                    applied.append((action, op))
                except Exception as exc:
                    logger.error(f"memory op failed ({op.slug}): {exc}")

            if applied:
                self._store.write_index_file()

        if applied:
            logger.info(f"memory: {len(applied)} operations ({', '.join(f'{a}:{o.slug}' for a, o in applied)})")

        return [op for _, op in applied]

    async def _apply_operation(self, op: MemoryOperation) -> str:
        """Apply a memory operation. Returns the action taken ('create' or 'update')."""
        now = datetime.now(timezone.utc)
        embedding = await self._llm.embed(f"{op.slug} {op.content}")
        existing = self._store.get(op.slug)

        if existing:
            self._store.update(op.slug, op.content, embedding=embedding)
            return "update"

        entry = MemoryEntry(
            slug=op.slug, type=op.type,
            content=op.content, created=now, updated=now,
        )
        self._store.create(entry, embedding=embedding)
        return "create"

    def _format_conversation(self, messages: list[ChatCompletionMessageParam]) -> str:
        lines = []
        for msg in messages[-self._cfg.memory_ctx_trunc_n:]:
            if content := msg.get("content", ""):
                role = msg.get("role", "").upper()
                lines.append(f"{role}: {str(content)[:self._cfg.memory_msg_max_len]}")
        return "\n".join(lines)
