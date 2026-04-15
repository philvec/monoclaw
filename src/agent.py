import asyncio
import json
from datetime import datetime, timezone

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from context import ContextManager
from llm import LLMClient
from memory import MemoryManager
from scheduler import CronJob
from channels import WebSocketChannelManager, InboundMessage
from config import CRON_CHANNEL, logger, SYSERR
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import ChatCompletionMessageToolCallParam
from tools import ToolRegistry

_MAX_TOOL_ITERATIONS = 20
_CHECKPOINT_PATH = Path("./data/history.jsonl")
_ARCHIVE_DIR = Path("./data/archive")

_INTENT_PROMPT = (
    "DECISION POINT (internal, not delivered to anyone). Based on the current inbound message, "
    "the conversation history, and the channel rules in your system prompt, decide: do you want "
    "to reply to the user on INPUT CHANNEL this turn? If yes, whatever you write as assistant "
    "content during the turn will be auto-delivered. If no, your content stays scratchpad. "
    "Return JSON per the schema. Keep `reason` short."
)


class TurnIntent(BaseModel):
    will_reply: bool = Field(
        description="True if this turn should produce a reply (auto-delivered content); False to stay silent"
    )
    reason: str = Field(default="", description="One short sentence (optional)")


class ReconsideredReply(BaseModel):
    reply_now: bool = Field(description="True if you want to send a message now; False to stay silent")
    text: str = Field(default="", description="Exact delivery text if reply_now=True; empty otherwise")
    reason: str = Field(default="", description="One short sentence explaining the decision")


class Session(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    history: list[ChatCompletionMessageParam] = []
    lock: asyncio.Lock = Field(default_factory=asyncio.Lock)


class AgentLoop:
    def __init__(
        self,
        llm: LLMClient,
        tool_registry: ToolRegistry,
        memory: MemoryManager,
        ctx: ContextManager,
        channel_manager: WebSocketChannelManager,
    ) -> None:
        self._llm = llm
        self._tool_registry = tool_registry
        self._memory = memory
        self._ctx = ctx
        self._channel_manager = channel_manager
        self._session = Session()

    async def startup(self) -> None:
        """Restore checkpoint and pre-warm the LLM cache."""
        try:
            self._session.history = self._restore_checkpoint()
        except Exception as exc:
            logger.error(f"failed to restore checkpoint on startup: {exc}")
        if self._session.history:
            logger.info(f"restored {len(self._session.history)} messages, warming cache")
            await self._warm_cache()

    # Public entry points

    async def handle_message(self, msg: InboundMessage) -> None:
        logger.info(f"acquiring session lock for {msg.channel}")
        async with self._session.lock:
            await self._process(msg)

    async def handle_cron(self, job: CronJob) -> None:
        synthetic = InboundMessage(channel=CRON_CHANNEL, text=job.message, timestamp=0)
        async with self._session.lock:
            await self._process(synthetic)

    async def _process(self, msg: InboundMessage) -> None:
        """Run one full agent turn: LLM call, tool loop, checkpoint, memory extraction."""
        # Assemble turn messages, starting with the system prompt
        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=self._memory.build_system_prompt(),
            )
        ]

        # Restore checkpoint from disk on first turn (skipped if startup() already ran)
        if not self._session.history:
            try:
                self._session.history = self._restore_checkpoint()
            except Exception as exc:
                logger.warning(err := f"failed to restore checkpoint: {exc}")
                messages.append(
                    ChatCompletionUserMessageParam(
                        role="user",
                        content=f"[{SYSERR} — running turn with empty history ({err})]",
                    )
                )
        messages.extend(self._session.history)

        # Inject channel context as a user note (system role not allowed mid-conversation)
        now = datetime.now(timezone.utc).astimezone()
        ctx_lines = [
            f"INPUT CHANNEL: {msg.channel}",
            f"CURRENT DATETIME: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        ]
        messages.append(ChatCompletionUserMessageParam(role="user", content="\n".join(ctx_lines)))
        user_msg = ChatCompletionUserMessageParam(role="user", content=msg.text)
        messages.append(user_msg)

        # Archive inbound message immediately so it survives any mid-turn failure
        self._session.history.append(user_msg)
        self._append_to_checkpoint(user_msg)

        # Intent decision: ask the model up-front whether this turn should produce a reply.
        # Cron turns skip this — they don't have a single inbound human to reply to.
        if msg.channel == CRON_CHANNEL:
            will_reply = False
            logger.info("turn from cron: will_reply=False")
        else:
            intent = await self._decide_intent(messages)
            will_reply = intent.will_reply
            logger.info(f"turn from {msg.channel!r}: will_reply={will_reply} ({intent.reason!r})")

        # As soon as we know a reply is coming, send an empty chunk so the bridge knows
        # something is on the way and can render a typing indicator during tool iterations.
        if will_reply and msg.channel != CRON_CHANNEL:
            try:
                await self._channel_manager.send_chunk(msg.channel, "")
            except Exception as exc:
                logger.warning(f"typing-chunk to {msg.channel!r} skipped: {exc}")

        # Tool execution loop
        iterations = 0

        while iterations < _MAX_TOOL_ITERATIONS:
            iterations += 1

            # Wire streaming auto-delivery when the model has committed to reply: each content
            # token goes to the inbound channel as it arrives, and an `end` frame closes the
            # message after the LLM call returns (if we streamed anything).
            streamed = False

            async def stream_delta(text: str) -> None:
                nonlocal streamed
                try:
                    await self._channel_manager.send_chunk(msg.channel, text)
                    streamed = True
                except Exception as exc:
                    logger.warning(f"auto-delivery chunk to {msg.channel!r} skipped: {exc}")

            on_delta = stream_delta if (will_reply and msg.channel != CRON_CHANNEL) else None

            logger.info(f"llm call start (iter={iterations}, msgs={len(messages)})")
            response = await self._llm.chat(messages, tools=self._tool_registry.definitions, on_delta=on_delta)
            self._ctx.update(response)

            if streamed:
                try:
                    await self._channel_manager.end_msg(msg.channel)
                except Exception as exc:
                    logger.warning(f"auto-delivery end-frame to {msg.channel!r} skipped: {exc}")

            if response.finish_reason == "error":
                logger.error(err := f"LLM error: {response.error or 'unknown'}")
                messages.append(
                    ChatCompletionUserMessageParam(
                        role="user", content=f"[{SYSERR} — no response generated this turn ({err})]"
                    )
                )
                break

            # Append assistant message
            assistant_msg = ChatCompletionAssistantMessageParam(role="assistant", content=response.content or "")
            if response.tool_calls:
                tool_call_list: list[ChatCompletionMessageToolCallParam] = []
                for tc in response.tool_calls:
                    try:
                        args_json = json.dumps(tc.arguments)
                    except (TypeError, ValueError) as exc:
                        logger.error(f"failed to serialize arguments for tool {tc.name!r}: {exc}")
                        args_json = "{}"
                    tool_call_list.append(
                        ChatCompletionMessageToolCallParam(
                            id=tc.id, type="function", function={"name": tc.name, "arguments": args_json}
                        )
                    )
                assistant_msg["tool_calls"] = tool_call_list
            messages.append(assistant_msg)

            if response.finish_reason != "tool_calls" or not response.tool_calls:
                break

            if iterations == _MAX_TOOL_ITERATIONS:
                # Synthesize error responses so the assistant's tool_calls are not orphaned
                # (the chat API rejects history containing tool_calls without paired tool results).
                logger.warning(f"tool iteration limit ({_MAX_TOOL_ITERATIONS}) reached — skipping final tool calls")
                for tc in response.tool_calls:
                    messages.append(
                        ChatCompletionToolMessageParam(
                            role="tool",
                            tool_call_id=tc.id,
                            content=f"error: tool iteration limit ({_MAX_TOOL_ITERATIONS}) reached — not executed",
                        )
                    )
                break

            # Execute all requested tools
            for tc in response.tool_calls:
                logger.info(f"executing tool {tc.name!r} args={tc.arguments!r}")
                result = await self._tool_registry.execute(tc.name, tc.arguments)
                messages.append(ChatCompletionToolMessageParam(role="tool", tool_call_id=tc.id, content=result))

        # Post-turn reconsideration on silent turns — give the model one last chance to change
        # its mind. (Auto-delivery only fires when will_reply=True, so reaching here with
        # will_reply=False means nothing has gone out to the inbound channel.)
        if not will_reply and msg.channel != CRON_CHANNEL:
            reconsider = await self._reconsider_silence(messages, msg.channel)
            logger.info(f"reconsider on {msg.channel!r}: reply_now={reconsider.reply_now} ({reconsider.reason!r})")
            if reconsider.reply_now and reconsider.text.strip():
                try:
                    await self._channel_manager.send_full_msg(msg.channel, reconsider.text)
                    messages.append(ChatCompletionAssistantMessageParam(role="assistant", content=reconsider.text))
                except Exception as exc:
                    logger.warning(f"reconsider delivery to {msg.channel!r} skipped: {exc}")

        # Persist turn to session history (drop only the base system prompt; rebuilt fresh each turn)
        self._session.history = messages[1:]
        self._save_checkpoint()

        # Fire-and-forget: compact history and extract memories
        asyncio.create_task(self._compact_session(), name="compact")
        asyncio.create_task(self._run_extract_memories(self._session.history), name="extract")

    # Intent / reconsider helpers

    async def _decide_intent(self, messages: list[ChatCompletionMessageParam]) -> TurnIntent:
        decision_msgs: list[ChatCompletionMessageParam] = [
            *messages,
            ChatCompletionUserMessageParam(role="user", content=_INTENT_PROMPT),
        ]
        resp = await self._llm.chat(decision_msgs, response_model=TurnIntent)
        if isinstance(resp.parsed, TurnIntent):
            return resp.parsed
        logger.warning(f"intent decision parse failed (finish={resp.finish_reason}); defaulting will_reply=True")
        return TurnIntent(will_reply=True, reason="parse-fail default")

    async def _reconsider_silence(self, messages: list[ChatCompletionMessageParam], channel: str) -> ReconsideredReply:
        prompt = (
            "RECONSIDERATION (internal). You initially chose to stay silent this turn. After any "
            f"internal processing, decide now: do you want to send a message on channel '{channel}' "
            "after all? If yes, provide the exact text the user will see (follow your brevity rules "
            "— minimal words). If no, confirm silence. Return JSON per the schema."
        )
        decision_msgs: list[ChatCompletionMessageParam] = [
            *messages,
            ChatCompletionUserMessageParam(role="user", content=prompt),
        ]
        resp = await self._llm.chat(decision_msgs, response_model=ReconsideredReply)
        if isinstance(resp.parsed, ReconsideredReply):
            return resp.parsed
        logger.warning(f"reconsider parse failed (finish={resp.finish_reason}); staying silent")
        return ReconsideredReply(reply_now=False, text="")

    # Checkpoint

    def _save_checkpoint(self) -> None:
        try:
            _CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            _CHECKPOINT_PATH.write_text("\n".join(json.dumps(m) for m in self._session.history) + "\n")
        except Exception as exc:
            logger.error(f"failed to save checkpoint: {exc}")

    def _restore_checkpoint(self) -> list[ChatCompletionMessageParam]:
        if not _CHECKPOINT_PATH.exists():
            return []
        entries = []
        for line in _CHECKPOINT_PATH.read_text().splitlines():
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning(f"skipping corrupted checkpoint line: {exc}")
        return entries

    def _append_to_checkpoint(self, msg: ChatCompletionMessageParam) -> None:
        try:
            with open(_CHECKPOINT_PATH, "a") as f:
                f.write(json.dumps(msg) + "\n")
        except Exception as exc:
            logger.error(f"failed to append to checkpoint: {exc}")

    async def _compact_session(self) -> None:
        if not self._ctx.should_compact():
            return
        async with self._session.lock:
            self._archive_checkpoint(self._session.history)
            self._session.history = await self._ctx.compact(
                self._session.history,
                self._llm,
                memory_flush_fn=self._memory.flush_memories,
            )
            self._save_checkpoint()
        # pre-warm the LLM cache with the compacted history
        asyncio.create_task(self._warm_cache(), name="warm-cache")

    async def _warm_cache(self) -> None:
        """Prefill the LLM cache with current history. Matches _process() message format."""
        try:
            messages: list[ChatCompletionMessageParam] = [
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=self._memory.build_system_prompt(),
                ),
                *self._session.history,
                ChatCompletionUserMessageParam(role="user", content="INPUT CHANNEL: warmup"),
                ChatCompletionUserMessageParam(role="user", content="."),
            ]
            await self._llm.chat(messages, tools=self._tool_registry.definitions, max_tokens=1)
            logger.info("cache pre-warmed")
        except Exception as exc:
            logger.error(f"cache warm-up failed: {exc}")

    def _archive_checkpoint(self, history: list[ChatCompletionMessageParam]) -> None:
        if not history:
            return
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        try:
            _ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
            (_ARCHIVE_DIR / f"{ts}.jsonl").write_text("\n".join(json.dumps(m) for m in history) + "\n")
        except Exception as exc:
            logger.error(f"failed to archive checkpoint: {exc}")

    async def _run_extract_memories(self, history: list[ChatCompletionMessageParam]) -> None:
        try:
            ops = await self._memory.extract_memories(history)
            if ops:
                self._append_to_checkpoint(
                    ChatCompletionUserMessageParam(
                        role="user",
                        content="[MEMORY SAVED]\n" + "\n".join(f"- {op.slug} ({op.type})" for op in ops),
                    ),
                )
        except Exception as exc:
            logger.error(err := f"memory extraction failed: {exc}")
            self._append_to_checkpoint(
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"[{SYSERR} — memory was NOT saved: ({err})]",
                ),
            )
