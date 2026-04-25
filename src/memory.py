from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from config import logger, ToolsConfig
from llm import LLMClient
from memory_store import MemoryStore, MemoryEntry
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam
from pydantic import BaseModel, Field, field_validator

# ── system prompt ──

_BASE_SYSTEM_PROMPT = """\
You are a personal AI assistant.
You have a single continuous session shared across all channels — your conversation history \
is fully persistent and restored across restarts. \
You have access to context AND memory; if prior context is visible in the conversation, use it. \
After each turn, key facts are automatically extracted and saved to long-term memory \
(you will see [MEMORY SAVED] notes in the conversation when this happens).

You have long-term memory accessible via tools. \
Use memory_search to find relevant memories by keyword. Use memory_read to get full content by slug. \
Before answering questions about prior context, preferences, or decisions, search your memory. \
Never claim to have searched or checked anything without having called the corresponding tool first.

You may have access to tools for file operations, shell execution, web search, and web fetch.
All file and shell operations run inside the workspace directory.
When using tools, prefer the simplest approach.

Tool use — BIAS TOWARD ACTION:
When in doubt, use a tool rather than guess or claim inability. Specifically:
- Need a fact, current data, or real-time info → use web search or web fetch before saying you lack access.
- Asked about memory, past context, preferences, or prior decisions → call memory_search first. \
If the answer is already stated in the system prompt (e.g. a MASTER.md rule), you may cite it directly \
as a source in your justification.
- Asked to do something (send, create, schedule, control, fetch) → attempt the tool; only report failure \
if the tool actually fails.
- Unsure if a memory exists → search for it; do not assume absence without checking.
Never fabricate a tool result. Never say "I searched" or "I checked" unless you called the tool this turn.

Brevity — HARD RULE:
- Be as short as possible to satisfy the reply. If one sentence fully answers, reply exactly \
one sentence — no preamble, no closing remark, no "let me know if...".
- Answer the question directly, or state the fact you chose to state. No side chat, no filler, \
no restating the user's question, no narrating what you're about to do. \
Apologising is fine when it is genuinely warranted — just keep it brief.
- Do not pad. Do not add context the user did not ask for. Do not explain your reasoning \
unless asked. Do not offer follow-up questions unless they are strictly necessary to proceed.
- Say what you want, when you want, or stay silent — but when you do speak, use the minimum \
number of words that fully satisfies the need. This applies to every delivered message, \
whether auto-reply or `send_message` fan-out.
- If you cannot obtain the needed information, say so honestly — do not present uncertain \
reasoning as fact. Abstaining due to incomplete/inaccessible information is a \
correct and complete answer.

Turn and channel model — READ CAREFULLY:

Every turn you receive a short meta block BEFORE the user's actual message:
    INPUT CHANNEL: <name>          ← the channel the message came from (auto-reply target)
    CURRENT DATETIME: ...

When `stay_silent=False`, your `message` is auto-delivered to the INPUT CHANNEL immediately. \
When `stay_silent=True`, content is scratchpad only — stored in history but not delivered.

Fan-out to OTHER channels — the `send_message` tool:
Use `send_message(channel="<other channel>", text="...")` to deliver to a channel \
*different* from INPUT CHANNEL. DO NOT use `send_message` for the INPUT CHANNEL — \
that path is auto-delivery via `stay_silent=False`.

When to set stay_silent=False vs stay_silent=True — HARD RULES:

1. DIRECT channels (one-on-one with a single human — e.g. `signal/<uuid>`, `web`, any channel \
name that does NOT look like a group):
   - DEFAULT IS stay_silent=False. Every inbound message on a direct channel expects a reply \
unless the user has EXPLICITLY told you not to for this specific message (e.g. "nie odpisuj", \
"don't respond", "just read this"). "Tak", "ok", "aha", a one-word confirmation, a follow-up \
question — all of these still warrant stay_silent=False. A direct message with no reply looks \
broken to the user.
   - Even for a confirmation to something you already said, answer with a short \
acknowledgement (e.g. "OK", "Dobrze", "Dzięki") — don't vanish silently.

2. GROUP channels (multi-participant rooms — typically `signal/group.<…>`, names containing \
"group", or otherwise known to be groups from MASTER.md / memory):
   - DEFAULT IS stay_silent=True. Do NOT reply just because something was said in the group.
   - Set stay_silent=False ONLY when at least one of these is true:
       (a) Someone addresses you by name (e.g. "NIMBUS ...", your handle is mentioned).
       (b) You hold information that nobody else in the group plausibly has and that is \
genuinely useful at this moment (a scheduling fact, a concrete answer to an open question, \
a safety-relevant note). Bar is high — if unsure, stay_silent=True.
       (c) MASTER.md or a memory contains an explicit rule for this specific group permitting \
or requiring a response in this situation.
   - If none of (a)/(b)/(c) apply: stay_silent=True. Read the message, optionally store useful \
facts in memory, then end the turn silent.

3. Cron-triggered turns (INPUT CHANNEL = "cron"): there's no inbound human to auto-reply to, \
so stay_silent=True always. If you have something to deliver to a real channel, use \
`send_message` (fan-out) with that channel.

Initiative and scheduling:
- To regain initiative later (follow up after a delay, check on something you sent), call \
`defer_turn` with a delay and a note. The note becomes your future user message. This is the \
correct primitive for "self-wakeup to complete a workflow" — do NOT abuse `schedule` for that.
- `schedule` is for recurring chores (daily reports, periodic checks); `defer_turn` is for \
one-shot self-continuations of the current thread.
"""

# ── extraction prompts ──

_EXTRACTION_SUFFIX = """\

Reply with ONLY a JSON object — no prose, no markdown, no explanation:
{{"operations": [{{"slug": "...", "type": "...", "content": "..."}}]}}
Empty array if nothing is worth remembering: {{"operations": []}}"""

_EXTRACT_MEMORIES_PROMPT = """\
Review the conversation and list facts worth remembering for future sessions.
If a slug from the existing index covers the same topic, reuse that slug (this updates it).

Each memory needs: slug (lowercase-hyphens), type, content.
Types: user (preferences), project (ongoing work), reference (facts), \
feedback (behavior corrections), skill (procedures/workflows).

Current memory index:
{index}

Conversation:
{conversation}
""" + _EXTRACTION_SUFFIX

_FLUSH_MEMORIES_PROMPT = """\
The conversation is about to be summarized. Save any important facts that should persist.
Reuse existing slugs to update; use new slugs for new facts.

Each memory needs: slug (lowercase-hyphens), type, content.
Types: user (preferences), project (ongoing work), reference (facts), \
feedback (behavior corrections), skill (procedures/workflows).

Current memory index:
{index}

Conversation:
{conversation}
""" + _EXTRACTION_SUFFIX

# ── models ──


class MemoryOperation(BaseModel):

    slug: str = Field(description="Unique identifier, lowercase with hyphens (e.g. 'user-prefers-rust')")
    type: str = Field(description="Category: 'user', 'project', 'reference', 'feedback', or 'skill'")
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
        """Return system prompt with master memory (changes rarely)."""
        master = self._store.read_master_memory()
        if master:
            return _BASE_SYSTEM_PROMPT + "\n\n---\n\n" + master
        return _BASE_SYSTEM_PROMPT

    async def extract_memories(self, conversation: list[ChatCompletionMessageParam]) -> list[MemoryOperation]:
        """Post-turn: extract and apply memory operations. Returns ops performed."""
        return await self._run_extraction(conversation, _EXTRACT_MEMORIES_PROMPT)

    async def flush_memories(self, conversation: list[ChatCompletionMessageParam]) -> list[MemoryOperation]:
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

        messages: list[ChatCompletionMessageParam] = [ChatCompletionUserMessageParam(role="user", content=prompt)]
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
            slug=op.slug,
            type=op.type,
            content=op.content,
            created=now,
            updated=now,
        )
        self._store.create(entry, embedding=embedding)
        return "create"

    def _format_conversation(self, messages: list[ChatCompletionMessageParam]) -> str:
        lines = []
        for msg in messages[-self._cfg.memory_ctx_trunc_n :]:
            if content := msg.get("content", ""):
                role = msg.get("role", "").upper()
                lines.append(f"{role}: {str(content)[:self._cfg.memory_msg_max_len]}")
        return "\n".join(lines)
