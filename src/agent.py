import asyncio
import json
from datetime import datetime, timezone

from collections.abc import Awaitable, Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from context import ContextManager
from llm import LLMClient, LLMResponse
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

    # Public entry points

    async def handle_message(self, msg: InboundMessage) -> None:
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

        # Restore checkpoint from disk on first turn
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

        # Always extend messages with accumulated history
        messages.extend(self._session.history)

        # Inject channel context as a user note (system role not allowed mid-conversation)
        expected_output = self._channel_manager.resolve_output(msg.channel)
        ctx_lines = [f"INPUT CHANNEL: {msg.channel}", f"CURRENTLY_SET_OUTPUT_CHANNEL: {expected_output}"]
        messages.append(ChatCompletionUserMessageParam(role="user", content="\n".join(ctx_lines)))
        user_msg = ChatCompletionUserMessageParam(role="user", content=msg.text)
        messages.append(user_msg)

        # Archive inbound message immediately so it survives any mid-turn failure
        self._session.history.append(user_msg)
        self._append_to_checkpoint(user_msg)

        # Tool execution loop
        response: LLMResponse | None = None
        iterations = 0
        on_delta_fn: Callable[[str], Awaitable[None]] | None = None

        async def _on_delta(content: str) -> None:
            nonlocal on_delta_fn
            if on_delta_fn is None:
                output = self._channel_manager.resolve_output(msg.channel)
                if output is not None:
                    on_delta_fn = self._channel_manager.make_on_delta(output)
            if on_delta_fn is not None:
                try:
                    await on_delta_fn(content)
                except Exception as exc:
                    logger.error(f"streaming delta failed: {exc}")

        while iterations < _MAX_TOOL_ITERATIONS:
            iterations += 1

            response = await self._llm.chat(messages, tools=self._tool_registry.definitions, on_delta=_on_delta)
            self._ctx.update(response)

            if response.finish_reason == "error":
                logger.error(err := f"LLM error: {response.error or 'unknown'}")
                messages.append(
                    ChatCompletionUserMessageParam(
                        role="user", content=f"[{SYSERR} — no response generated this turn ({err})]"
                    )
                )
                break

            # Append assistant message
            tool_call_list: list[ChatCompletionMessageToolCallParam] = []
            if response.tool_calls:
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
            if tool_call_list:
                assistant_msg: ChatCompletionMessageParam = ChatCompletionAssistantMessageParam(
                    role="assistant", content=response.content or "", tool_calls=tool_call_list
                )
            else:
                assistant_msg = ChatCompletionAssistantMessageParam(role="assistant", content=response.content or "")
            messages.append(assistant_msg)

            if response.finish_reason != "tool_calls" or not response.tool_calls:
                break

            if iterations == _MAX_TOOL_ITERATIONS:
                logger.warning(f"tool iteration limit ({_MAX_TOOL_ITERATIONS}) reached — skipping final tool calls")
                break

            # Execute all requested tools
            for tc in response.tool_calls:
                logger.info(f"executing tool {tc.name!r} args={tc.arguments!r}")
                result = await self._tool_registry.execute(tc.name, tc.arguments)
                messages.append(ChatCompletionToolMessageParam(role="tool", tool_call_id=tc.id, content=result))

        # Deliver response via output channel
        output = self._channel_manager.resolve_output(msg.channel)
        if output is not None and response is not None:
            logger.info(f"using output: channel: {output}")
            try:
                await self._channel_manager.end_message(output)
            except Exception as exc:
                logger.error(f"failed to deliver end signal: {exc}")
        self._channel_manager.reset_output()

        # Persist turn to session history (drop only the base system prompt; rebuilt fresh each turn)
        self._session.history = messages[1:]
        self._save_checkpoint()

        # Fire-and-forget: compact history and extract memories
        asyncio.create_task(self._compact_session(), name="compact")
        asyncio.create_task(self._run_extract_memories(self._session.history), name="extract")

    # Checkpoint

    def _save_checkpoint(self) -> None:
        try:
            _CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            _CHECKPOINT_PATH.write_text("\n".join(json.dumps(m) for m in self._session.history))
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
                self._session.history, self._llm,
                memory_flush_fn=self._memory.flush_memories,
            )
            self._save_checkpoint()

    def _archive_checkpoint(self, history: list[ChatCompletionMessageParam]) -> None:
        if not history:
            return
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        try:
            _ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
            (_ARCHIVE_DIR / f"{ts}.jsonl").write_text("\n".join(json.dumps(m) for m in history))
        except Exception as exc:
            logger.error(f"failed to archive checkpoint: {exc}")

    async def _run_extract_memories(self, history: list[ChatCompletionMessageParam]) -> None:
        try:
            ops = await self._memory.extract_memories(history)
            if ops:
                self._append_to_checkpoint(
                    ChatCompletionUserMessageParam(
                        role="user",
                        content="[MEMORY SAVED]\n" + "\n".join(
                            f"- {op.slug} ({op.type})" for op in ops
                        ),
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
