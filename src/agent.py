import asyncio
import json
import random
from datetime import datetime, timezone

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from context import ContextManager
from llm import LLMClient
from memory import MemoryManager
from models import Answer, Review
from reviewer import Reviewer, MAX_NEGATIVE_REVIEWS
from scheduler import CronJob
from channels import WebSocketChannelManager, InboundMessage
from config import ARCHIVE_DIR, CRON_CHANNEL, logger, MAX_STORED_MSG_CHARS, SYSERR
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
_MAX_EXTRACT_CANCELS = 8  # force extraction after this many consecutive deferrals
_CHECKPOINT_PATH = Path("./data/history.jsonl")

_MAX_REVIEWS_FALLBACK_MESSAGES = [
    "Sorry, I couldn't produce a coherent response for this one. 🤷",
    "I have to pass on this one — couldn't get to a verified answer. 🙈",
    "I have to abstain here, sorry. 🫣",
    "Can't give you a solid answer on this — I'll have to skip it. 😬",
    "Sorry, no coherent result from me on this. 😅",
    "I'll have to sit this one out — couldn't verify my response. 🪑",
]

_SCHEMA_INSTRUCTIONS = (
    "RESPONSE SCHEMA RULES:\n"
    "- justification: internal reasoning only, never shown to the user. "
    "Must explicitly justify BOTH the stay_silent decision AND every factual claim put in the message. "
    "For each claim, cite the exact source: system prompt / MASTER.md rule (e.g. 'system prompt states X'), "
    "named tool result (e.g. 'memory_search returned empty'), "
    "named memory entry (e.g. 'memory user-prefers-polish'), quoted past message, or exact channel rule. "
    "For stay_silent=True: quote the specific rule or exact user instruction that permits silence — "
    "e.g. 'channel rule: group channels default to silent' or 'user said \"nie odpisuj\" in message at T'. "
    "For any admission of inability (e.g. 'I don't have that data', 'I found nothing'): cite the tool "
    "you ran and what it returned, AND the rule that directs you to inform the user of this. "
    "Vague justifications ('seemed appropriate', 'no relevant info') will fail review.\n"
    "- message: the exact text delivered to the user. Write ONLY the direct answer — no preamble, "
    "no follow-up offers ('Is there anything else?'), no meta-commentary about what you are doing, "
    "no unsolicited suggestions. Unless the user explicitly asked for those, leave them out. "
    "If stay_silent=False, message MUST be non-empty — an empty message with stay_silent=False is invalid.\n"
    "- stay_silent: True to stay silent (internal work only); False to deliver message.\n"
    "Every response is reviewed. The reviewer verifies that every claim in the message and the "
    "stay_silent decision are each traceable to a specific cited source in the justification."
)


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
        self._reviewer = Reviewer(llm)
        llm.set_schema_tools(tool_registry.definitions)
        self._session = Session()
        self._foreground_count = 0
        self._foreground_idle = asyncio.Event()
        self._foreground_idle.set()
        self._pending_extract: asyncio.Task | None = None
        self._extract_cancel_count = 0
        self._pending_warm_reviewer: asyncio.Task | None = None

    def _build_system_prompt(self) -> str:
        return self._memory.build_system_prompt() + "\n\n" + _SCHEMA_INSTRUCTIONS

    async def startup(self) -> None:
        """Restore checkpoint and pre-warm the LLM cache."""
        try:
            self._session.history = self._restore_checkpoint()
        except Exception as exc:
            logger.error(f"failed to restore checkpoint on startup: {exc}")
        if self._session.history:
            logger.info(f"♻️ restored {len(self._session.history)} messages, warming cache")
            await self._warm_cache()
            await self._warm_reviewer_cache()

    # Public entry points

    async def handle_message(self, msg: InboundMessage) -> None:
        logger.info(f"🔒 acquiring session lock for {msg.channel!r}")
        self._foreground_count += 1
        self._foreground_idle.clear()
        try:
            async with self._session.lock:
                await self._process(msg)
        finally:
            self._foreground_count -= 1
            if self._foreground_count == 0:
                self._foreground_idle.set()

    async def handle_cron(self, job: CronJob) -> None:
        synthetic = InboundMessage(channel=CRON_CHANNEL, text=job.message, timestamp=0)
        self._foreground_count += 1
        self._foreground_idle.clear()
        try:
            async with self._session.lock:
                await self._process(synthetic)
        finally:
            self._foreground_count -= 1
            if self._foreground_count == 0:
                self._foreground_idle.set()

    async def _process(self, msg: InboundMessage) -> None:
        """Run one full agent turn: LLM call, tool loop, checkpoint, memory extraction."""
        # Assemble turn messages, starting with the system prompt
        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=self._build_system_prompt(),
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

        # Tool execution loop
        iterations = 0
        llm_ok = False
        turn_delivered = False
        typing_signaled = False
        review_rejections = 0
        review_start_idx = -1  # index in messages where first Answer was appended
        review_accepted = False

        while iterations < _MAX_TOOL_ITERATIONS:
            iterations += 1

            logger.info(f"🤖 LLM call start (iter={iterations}, msgs={len(messages)})")
            response = await self._llm.chat(messages, tools=self._tool_registry.definitions, response_model=Answer)
            self._ctx.update(response)

            if response.finish_reason != "error":
                llm_ok = True
            if response.finish_reason == "error":
                logger.error(err := f"LLM error: {response.error or 'unknown'}")
                messages.append(
                    ChatCompletionUserMessageParam(
                        role="user", content=f"[{SYSERR} — no response generated this turn ({err})]"
                    )
                )
                break

            initial_answer: Answer | None = response.parsed if isinstance(response.parsed, Answer) else None
            initial_content = (
                response.content or ""
                if initial_answer is None
                else ("[STAYED SILENT]" if initial_answer.stay_silent else initial_answer.message)
            )
            if len(initial_content) > MAX_STORED_MSG_CHARS:
                logger.warning(f"response truncated ({len(initial_content)} chars, finish={response.finish_reason!r})")
                initial_content = initial_content[:MAX_STORED_MSG_CHARS] + "… [truncated]"
            assistant_msg = ChatCompletionAssistantMessageParam(role="assistant", content=initial_content)
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

            # Reviewer runs on Answer turns (no tool_calls). Tool-call iterations have parsed=None;
            # mixing reviewer with tool_call assistant messages would orphan tool results in history.
            if initial_answer is not None and not response.tool_calls and msg.channel != CRON_CHANNEL:
                if not typing_signaled and not initial_answer.stay_silent:
                    try:
                        await self._channel_manager.send_chunk(msg.channel, "")
                        typing_signaled = True
                    except Exception as exc:
                        logger.warning(f"typing signal to {msg.channel!r} failed: {exc}")

                if not initial_answer.stay_silent and not initial_answer.message.strip():
                    review = Review(
                        is_correct=False,
                        to_be_fixed=[
                            "stay_silent=False but message is empty — provide a non-empty message or set stay_silent=True."
                        ],
                    )
                else:
                    review = await self._reviewer.run_review(messages, assistant_msg)

                if review_start_idx < 0:
                    review_start_idx = len(messages)
                messages.append(assistant_msg)

                if review.is_correct:
                    review_accepted = True
                    logger.info(f"✅ review passed (attempt {review_rejections + 1})")
                    if not initial_answer.stay_silent and initial_answer.message.strip():
                        preview = initial_answer.message[:120] + ("…" if len(initial_answer.message) > 120 else "")
                        logger.info(f"📤 delivering to {msg.channel!r}: {preview!r}")
                        try:
                            await self._channel_manager.send_full_msg(msg.channel, initial_answer.message)
                            turn_delivered = True
                        except Exception as exc:
                            logger.warning(f"delivery to {msg.channel!r} skipped: {exc}")
                    else:
                        logger.info(f"🤫 staying silent on {msg.channel!r}")
                        turn_delivered = True
                        if typing_signaled:
                            try:
                                await self._channel_manager.end_msg(msg.channel)
                            except Exception as exc:
                                logger.warning(f"end signal to {msg.channel!r} skipped: {exc}")
                    break

                review_rejections += 1
                reviewer_content = "[REVIEW — is_correct=False]"
                if review.to_be_fixed:
                    reviewer_content += "\n" + "\n".join(f"- {p}" for p in review.to_be_fixed)
                reviewer_content += (
                    "\nRetry with a corrected response. "
                    "All three fields are required: justification (str), message (str), stay_silent (bool). "
                    "If stay_silent=False, message MUST be non-empty."
                )
                logger.info(
                    f"❌ review rejected (attempt {review_rejections}/{MAX_NEGATIVE_REVIEWS}): {review.to_be_fixed}"
                )
                messages.append(ChatCompletionUserMessageParam(role="user", content=reviewer_content))

                if review_rejections >= MAX_NEGATIVE_REVIEWS:
                    logger.warning(f"🚫 max negative reviews ({MAX_NEGATIVE_REVIEWS}) reached, suppressing reply")
                    self._reviewer.archive_trail(messages[review_start_idx:])
                    fallback = random.choice(_MAX_REVIEWS_FALLBACK_MESSAGES)
                    logger.warning(f"sending fallback to {msg.channel!r}: {fallback!r}")
                    try:
                        await self._channel_manager.send_full_msg(msg.channel, fallback)
                    except Exception as exc:
                        logger.warning(f"fallback delivery to {msg.channel!r} skipped: {exc}")
                    turn_delivered = True
                    break

                continue  # reviewer rejected — agent gets full next iteration with tools available
            else:
                messages.append(assistant_msg)
                if not response.tool_calls and msg.channel != CRON_CHANNEL:
                    logger.warning(f"parse failure on {msg.channel!r} (iter={iterations}), retrying")
                    messages.append(
                        ChatCompletionUserMessageParam(
                            role="user",
                            content=f"[{SYSERR} — your response did not parse as a valid Answer. "
                            "Retry with a valid JSON object: justification (str), message (str), stay_silent (bool).]",
                        )
                    )
                    continue

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
            if msg.channel != CRON_CHANNEL:
                try:
                    await self._channel_manager.send_chunk(msg.channel, "")
                    typing_signaled = True
                except Exception as exc:
                    logger.warning(f"typing signal to {msg.channel!r} failed: {exc}")
            for tc in response.tool_calls:
                logger.info(f"🔧 tool call: {tc.name!r}")
                result = await self._tool_registry.execute(tc.name, tc.arguments)
                messages.append(ChatCompletionToolMessageParam(role="tool", tool_call_id=tc.id, content=result))

        # Safety net: if the loop exhausted retries on parse failures, client is still waiting
        if not turn_delivered and msg.channel != CRON_CHANNEL:
            fallback = random.choice(_MAX_REVIEWS_FALLBACK_MESSAGES)
            logger.warning(f"parse retries exhausted on {msg.channel!r}, sending fallback: {fallback!r}")
            try:
                await self._channel_manager.send_full_msg(msg.channel, fallback)
            except Exception as exc:
                logger.warning(f"parse-exhaustion fallback delivery to {msg.channel!r} skipped: {exc}")

        # Persist turn to session history.
        # Strip review trial: keep only the final accepted Answer (or a suppression note for MAX).
        # Pre-review messages (ctx, user, tool calls) are always kept intact.
        if review_start_idx >= 0:
            pre_review = messages[1:review_start_idx]
            if not review_accepted:
                review_outcome: list[ChatCompletionMessageParam] = [
                    ChatCompletionUserMessageParam(
                        role="user",
                        content=f"[REPLY SUPPRESSED — reviewer rejected after {review_rejections} attempt(s)]",
                    )
                ]
            else:
                final_msg = next(
                    (
                        m
                        for m in reversed(messages[review_start_idx:])
                        if m.get("role") == "assistant" and not m.get("tool_calls")
                    ),
                    None,
                )
                review_outcome = [final_msg] if final_msg else []
            self._session.history = pre_review + review_outcome
        else:
            self._session.history = messages[1:]
        self._save_checkpoint()

        # Fire-and-forget: compact history and extract memories (skip if LLM was down this turn)
        if llm_ok:
            asyncio.create_task(self._compact_session(), name="compact")
        if llm_ok and msg.channel != CRON_CHANNEL:
            if self._pending_warm_reviewer and not self._pending_warm_reviewer.done():
                self._pending_warm_reviewer.cancel()
            self._pending_warm_reviewer = asyncio.create_task(self._warm_reviewer_cache(), name="warm-reviewer-cache")

        # Coalesce extraction tasks. On cap hit, bypass the task system entirely and
        # await extraction directly — session lock is still held, so all new turns queue
        # behind us until it completes. No cancellation possible.
        if llm_ok:
            if self._pending_extract and not self._pending_extract.done():
                self._extract_cancel_count += 1
                self._pending_extract.cancel()
                self._pending_extract = None
                if self._extract_cancel_count >= _MAX_EXTRACT_CANCELS:
                    logger.info(
                        f"extract cap reached after {self._extract_cancel_count} deferrals, "
                        "blocking turn release until extraction completes"
                    )
                    self._extract_cancel_count = 0
                    await self._run_extract_memories(self._session.history, force=True)
                else:
                    logger.info(f"extract deferred (deferral #{self._extract_cancel_count}), new turn took priority")
                    self._pending_extract = asyncio.create_task(
                        self._run_extract_memories(self._session.history),
                        name="extract",
                    )
            else:
                self._extract_cancel_count = 0
                self._pending_extract = asyncio.create_task(
                    self._run_extract_memories(self._session.history),
                    name="extract",
                )

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
                msg = json.loads(line)
                content = msg.get("content")
                if isinstance(content, str) and len(content) > MAX_STORED_MSG_CHARS:
                    logger.warning(f"checkpoint: truncating oversized {msg.get('role')} message ({len(content)} chars)")
                    msg["content"] = content[:MAX_STORED_MSG_CHARS] + "… [truncated]"
                entries.append(msg)
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
            # Cancel inside the lock so no new turn can slip in and create a replacement
            # extract task between the cancellation and the start of flush_memories.
            if self._pending_extract and not self._pending_extract.done():
                self._pending_extract.cancel()
                self._pending_extract = None
                self._extract_cancel_count = 0
                logger.info("🗜️ extract task cancelled: compaction flush covers it")
            self._archive_checkpoint(self._session.history)
            self._session.history = await self._ctx.compact(
                self._session.history,
                self._llm,
                memory_flush_fn=self._memory.flush_memories,
            )
            self._save_checkpoint()
        # pre-warm the LLM cache with the compacted history
        asyncio.create_task(self._warm_cache(), name="warm-cache")
        if self._pending_warm_reviewer and not self._pending_warm_reviewer.done():
            self._pending_warm_reviewer.cancel()
        self._pending_warm_reviewer = asyncio.create_task(self._warm_reviewer_cache(), name="warm-reviewer-cache")

    async def _warm_cache(self) -> None:
        """Prefill the LLM cache with current history. Matches _process() message format."""
        try:
            messages: list[ChatCompletionMessageParam] = [
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=self._build_system_prompt(),
                ),
                *self._session.history,
                ChatCompletionUserMessageParam(role="user", content="INPUT CHANNEL: warmup"),
                ChatCompletionUserMessageParam(role="user", content="."),
            ]
            logger.info(f"🔥 warming agent cache ({len(messages)} msgs)")
            await self._llm.chat(messages, tools=self._tool_registry.definitions, max_tokens=1)
            logger.info("✅ agent cache warmed")
        except Exception as exc:
            logger.error(f"agent cache warm-up failed: {exc}")

    async def _warm_reviewer_cache(self) -> None:
        """Prefill the reviewer KV cache with the current history prefix."""
        try:
            if not self._foreground_idle.is_set():
                logger.info("⏳ reviewer cache warm-up: waiting for foreground idle")
            await self._foreground_idle.wait()
            messages: list[ChatCompletionMessageParam] = [
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=self._build_system_prompt(),
                ),
                *self._session.history,
            ]
            logger.info(f"🔥 warming reviewer cache ({len(messages)} msgs)")
            await self._reviewer.warm_cache(messages)
            logger.info("✅ reviewer cache warmed")
        except Exception as exc:
            logger.error(f"reviewer cache warm-up failed: {exc}")

    def _archive_checkpoint(self, history: list[ChatCompletionMessageParam]) -> None:
        if not history:
            return
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        try:
            ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
            (ARCHIVE_DIR / f"{ts}.jsonl").write_text("\n".join(json.dumps(m) for m in history) + "\n")
        except Exception as exc:
            logger.error(f"failed to archive checkpoint: {exc}")

    async def _run_extract_memories(self, history: list[ChatCompletionMessageParam], *, force: bool = False) -> None:
        if not force:
            try:
                await self._foreground_idle.wait()
            except asyncio.CancelledError:
                logger.info("❌ extract task cancelled while waiting for foreground idle")
                raise
            logger.info("🧠 extract triggered: foreground idle")
        else:
            logger.info("🧠 extract triggered: forced inline (cap reached)")
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
