import asyncio
import base64
import json
import mimetypes
import random
import re
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
from config import (
    ARCHIVE_DIR,
    CRON_CHANNEL,
    IMAGE_HISTORY_TURNS,
    IMAGE_MIME_EXT,
    IMAGES_DIR,
    LLM_IMAGE_MIMES,
    OUTBOUND_IMAGES_MAX_TOTAL_BYTES,
    sniff_image_mime,
    to_decodable_image,
    logger,
    MAX_STORED_MSG_CHARS,
    SYSERR,
)
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import ChatCompletionMessageToolCallParam
from tools import ToolRegistry

_MAX_TOOL_ITERATIONS = 20
_REPEATED_CALL_LIMIT = 2  # identical (name, args) executions per turn before the call is refused
_TOOL_CALLS_PER_TURN_LIMIT = 6  # per tool NAME — catches loops that vary one argument to evade the above
_MAX_EXTRACT_CANCELS = 8  # force extraction after this many consecutive deferrals
_CHECKPOINT_PATH = Path("./data/history.jsonl")

# Images never enter session history: _process_inner assigns history straight from the prompt list
# (see "Persist turn to session history" below), so content parts placed in `messages` would be
# written back to history and onto disk. History therefore stays str-only and carries markers;
# _with_images expands them into content parts at the llm.chat() boundary only, on a copy.
IMAGE_MARKER_PREFIX = "[IMAGE "
_IMAGE_MARKER = re.compile(r"^\[IMAGE ([^\s\]]+) (image/[^\s\]]+)\]$")
MAX_OUTBOUND_ATTACHMENTS = 4


def _persist_user_content(msg: InboundMessage) -> str:
    """Write inbound images to IMAGES_DIR; return the content stored in history: one
    '[IMAGE <file> <mime>]' marker per image, then the user's text. Text-only messages
    are returned verbatim. Raises on undecodable base64 or an unwritable directory."""
    if not msg.images:
        return msg.text
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    markers: list[str] = []
    notes: list[str] = []
    for i, img in enumerate(msg.images):
        raw = base64.b64decode(img.data, validate=True)
        try:
            # Convert here, not at read time: storing something the backend cannot decode would
            # make every later turn fail, not just this one.
            data, mime = to_decodable_image(raw, img.mime)
        except Exception as exc:
            logger.warning(f"undecodable inbound image from {msg.channel!r} ({img.mime}): {exc}")
            notes.append(f"[obrazek {img.name or i} w nieobsługiwanym formacie ({img.mime}) — nie widzę go]")
            continue
        fname = f"{msg.timestamp}-{i}{IMAGE_MIME_EXT.get(mime) or '.img'}"
        (IMAGES_DIR / fname).write_bytes(data)
        markers.append(f"{IMAGE_MARKER_PREFIX}{fname} {mime}]")
    logger.info(f"🖼️ stored {len(markers)} image(s) from {msg.channel!r} in {IMAGES_DIR}")
    return "\n".join(markers + notes + ([msg.text] if msg.text else []))


def _expand_markers(content: str) -> list[ChatCompletionContentPartParam]:
    """Leading '[IMAGE ...]' marker lines → image parts; the remaining text → one text part."""
    parts: list[ChatCompletionContentPartParam] = []
    rest: list[str] = []
    for line in content.split("\n"):
        if rest or not (m := _IMAGE_MARKER.match(line)):
            rest.append(line)
            continue
        # basename only: marker text is attacker-reachable (a user can type one, and the agent can
        # write one via its file tools), so never let it escape IMAGES_DIR — cf. tools._safe_path.
        data = (IMAGES_DIR / Path(m.group(1)).name).read_bytes()
        parts.append(
            ChatCompletionContentPartImageParam(
                type="image_url",
                image_url={"url": f"data:{m.group(2)};base64,{base64.b64encode(data).decode('ascii')}"},
            )
        )
    if text := "\n".join(rest).strip():
        parts.append(ChatCompletionContentPartTextParam(type="text", text=text))
    return parts


def _resolve_attachments(names: list[str]) -> tuple[list[str], list[str]]:
    """Answer.attachments → (marker lines, rejected names).

    The model supplies these filenames, so treat them as untrusted: basename only (never escape
    IMAGES_DIR) and the file must already exist there — i.e. it must be a picture the model was
    actually shown this conversation, not one it invented.
    """
    markers: list[str] = []
    rejected: list[str] = []
    budget = OUTBOUND_IMAGES_MAX_TOTAL_BYTES
    for raw in names[:MAX_OUTBOUND_ATTACHMENTS]:
        # tolerate the model echoing a whole "[IMAGE <file> <mime>]" marker instead of the filename
        m = _IMAGE_MARKER.match(raw.strip())
        fname = Path(m.group(1) if m else raw.strip()).name
        path = IMAGES_DIR / fname
        if fname and not path.is_file():
            # read_image/download_image store a timestamp-prefixed copy ("<stamp>-black_square.png"),
            # but the model naturally attaches the name it knows ("black_square.png"). Accept that
            # and resolve to the newest stored copy rather than rejecting a picture it really saw.
            matches = sorted(p.name for p in IMAGES_DIR.glob(f"*-{fname}") if p.is_file())
            if matches:
                fname = matches[-1]
                path = IMAGES_DIR / fname
        if not fname or not path.is_file():
            rejected.append(f"{raw} (no such image — you may have invented the filename)")
            continue
        mime = mimetypes.guess_type(fname)[0] or ""
        if mime not in LLM_IMAGE_MIMES:
            mime = sniff_image_mime(path.read_bytes()[:16])
        # Guard for stragglers stored before conversion existed: the reviewer has to render this,
        # and an undecodable one would 400 the whole review.
        if mime not in LLM_IMAGE_MIMES:
            rejected.append(f"{raw} (unsupported image format)")
            continue
        size = path.stat().st_size
        if size > budget:
            # base64 inflates by 4/3, and the whole answer travels as ONE websocket frame: exceeding
            # the negotiated limit kills the channel mid-delivery instead of just dropping a picture.
            rejected.append(f"{raw} (too large: {size / 1e6:.1f} MB, over the per-message budget)")
            continue
        budget -= size
        markers.append(f"{IMAGE_MARKER_PREFIX}{fname} {mime}]")
    return markers, rejected


def _request_preview(text: str, limit: int = 2000) -> str:
    """One-line form of the inbound request, for the 'answer THIS' markers. Group turns carry a
    whole transcript, so cap it."""
    one_line = " ".join((text or "").split()) or "(no text)"
    return one_line if len(one_line) <= limit else one_line[:limit] + "… [truncated]"


def _available_attachments(messages: list[ChatCompletionMessageParam]) -> list[str]:
    """Marker filenames present in this turn — the only names that may legitimately be attached.
    Used to make a rejection actionable: the model reliably invents a descriptive filename
    ('wawel-castle-krakow.jpg') instead of copying the opaque stored one, so tell it the real ones."""
    seen: list[str] = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, str):
            continue
        for line in content.split("\n"):
            if (mm := _IMAGE_MARKER.match(line)) and mm.group(1) not in seen:
                seen.append(mm.group(1))
    return seen


def _load_attachments(markers: list[str]) -> list[dict]:
    """Marker lines → wire payloads for the outbound frame."""
    out: list[dict] = []
    for line in markers:
        m = _IMAGE_MARKER.match(line)
        if m is None:
            continue
        data = (IMAGES_DIR / Path(m.group(1)).name).read_bytes()
        out.append({
            "mime": m.group(2),
            "data": base64.b64encode(data).decode("ascii"),
            "name": m.group(1),
        })
    return out


def _with_images(messages: list[ChatCompletionMessageParam]) -> list[ChatCompletionMessageParam]:
    """Prompt-time only: expand the newest IMAGE_HISTORY_TURNS marker-bearing messages into content
    parts. Returns a NEW list — `messages` and session history must stay str-only.

    Tool results are expanded too: the Qwen chat template renders a tool message inside a user block,
    so images are legal there, which is how read_image gets its picture into the context.
    """
    idxs = [
        i
        for i, m in enumerate(messages)
        if m.get("role") in ("user", "tool")
        and isinstance(c := m.get("content"), str)
        and c.startswith(IMAGE_MARKER_PREFIX)
    ]
    out = list(messages)
    expanded = 0
    for i in reversed(idxs):
        if expanded >= IMAGE_HISTORY_TURNS:
            break
        try:
            parts = _expand_markers(str(messages[i]["content"]))
            out[i] = {**messages[i], "content": parts}  # type: ignore[typeddict-item]
        except OSError as exc:
            # The marker is permanent history; a pruned or lost file must not brick every later turn.
            # The model still sees the marker line, so nothing is silently fabricated.
            logger.warning(f"image file missing for history[{i}], sending marker text only: {exc}")
            continue
        expanded += 1
    return out


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
    "Must explicitly justify every factual claim put in the message. "
    "For each claim, cite the exact source: system prompt / MASTER.md rule (e.g. 'system prompt states X'), "
    "named tool result (e.g. 'memory_search returned empty'), "
    "named memory entry (e.g. 'memory user-prefers-polish'), quoted past message, exact channel rule, "
    "or an image you were shown (a '[IMAGE ...]' message; cite as 'attached image shows X'). "
    "A message starting with '[IMAGE <file> <mime>]' has that picture ATTACHED — you can already see it, "
    "so describe it directly — that filename is an internal reference, not a workspace file.\n"
    "- SENDING A PICTURE: `attachments` is the only way. Get the picture into your context first "
    "(image_search → download_image for the web, read_image for a workspace file), check it is really "
    "what was asked for, then put that filename in `attachments`. A URL in the message text sends nothing. "
    "For any admission of inability (e.g. 'I don't have that data', 'I found nothing'): cite the tool "
    "you ran and what it returned, AND the rule that directs you to inform the user of this. "
    "Vague justifications ('seemed appropriate', 'no relevant info') will fail review.\n"
    "- message: the exact text delivered to the user. Write ONLY the direct answer — no preamble, "
    "no follow-up offers ('Is there anything else?'), no unsolicited suggestions. "
    "Unless the user explicitly asked for those, leave them out. Narration of what you are doing does not "
    "belong here — it goes in the interim line below. "
    "message reports what you ACTUALLY DID and found; it must never announce or promise work "
    "('Szukam w sieci…', 'zaraz sprawdzę', 'a potem napiszę skrypt'). If an action is still needed, call the "
    "tool FIRST and answer with its result — a message that narrates an action you did not perform this turn "
    "is a fabrication and will be rejected. "
    "message MUST NEVER be empty. Every turn gets an answer — there is no way to stay silent. If you could "
    "not do or find what was asked, say exactly that; if you have only a partial result, deliver it. An "
    "empty message is always a bug, never a choice.\n"
    "INTERIM LINE — REQUIRED BEFORE EVERY TOOL CALL: before you use any tool, say what you are about to "
    "do in exactly ONE short line of plain text (e.g. 'Szukam w sieci…', 'Piszę skrypt do policzenia "
    "tego…'). The user sees it straight away. Applies to EVERY tool — searches, file writes, run_command "
    "commands, memory reads alike. Exactly one line, never two, never the final answer — every time you "
    "reach for a tool, not just the slow ones. Do NOT use send_message for this; it is fan-out only and "
    "is refused for the input channel.\n"
    "Every response is reviewed. The reviewer verifies that every claim in the message is traceable to a "
    "specific cited source in the justification.\n"
    "TOOL POLICY: For questions about specific named entities (people, places, organisations, events), "
    "current facts (prices, schedules, availability, rankings), or any knowledge not confirmed by memory — "
    "if memory_search finds nothing relevant, call tools__web_search before answering. "
    "Never rely on training knowledge alone for such facts; it may be stale or wrong.\n"
    "ACTION RULE: Any action that requires a tool (editing a file, writing a file, running a command) "
    "MUST be executed via a tool call BEFORE your final Answer. "
    "Never state in your message that you edited, modified, updated, or wrote a file unless "
    "an edit_file or write_file tool result already exists in this turn's conversation. "
    "If the reviewer rejects your Answer for a missing tool call, your very next response "
    "MUST call that tool — simply rewording the claim without calling the tool will be rejected again."
)

_CLASSIFIER_INSTRUCTIONS = (
    "FAST PRE-AGENT CLASSIFIER:\n"
    "Before a message reaches you, every WebSocket message first passes through a fast, local "
    "classification layer (a small Qwen3.5-2B model). It returns structured output with a fixed, "
    "immutable schema: response_mode ('immediate' or 'complex'), output (text), and — when tools are "
    "configured — an optional tool_call.\n"
    "- complex → the layer does nothing; the message reaches you normally, as if it weren't there.\n"
    "- immediate → the layer answers the user directly with `output` and does NOT invoke you this turn; "
    "it still records the turn (user message + reply) in history so you know what was already done. In "
    "immediate mode the layer may also execute a whitelisted MCP tool (e.g. light control) and record a "
    "confirmation of the action in history.\n"
    "The classifier's behaviour is driven SOLELY by its own system prompt — a plain text file at "
    "data/memory/fast_classifier_system.md (same directory as MASTER.md). The file is re-read on every "
    "message, so your edits take effect immediately, without a restart. The output schema is hard-coded "
    "and immutable — do NOT try to change it. To change the fast classifier's behaviour, edit "
    "data/memory/fast_classifier_system.md with your file tools; when the user asks for such a change, "
    "do it that way.\n"
    "If you see a '[FAST CLASSIFIER ERROR: ...]' message in a turn, the layer hit an error (bad output "
    "format, model error, missing prompt file). The user's message still reached you — treat it as a "
    "signal that the classifier's prompt or configuration needs fixing."
)


def build_channel_ctx(channel: str) -> str:
    """Channel + current-datetime note injected ahead of the user's message.

    Built at the dispatch layer (main.on_message / handle_cron) and passed via the
    ``preamble`` parameter, rather than deep inside the turn assembly."""
    now = datetime.now(timezone.utc).astimezone()
    return f"INPUT CHANNEL: {channel}\nCURRENT DATETIME: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}"


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
        return (
            self._memory.build_system_prompt()
            + "\n\n"
            + _SCHEMA_INSTRUCTIONS
            + "\n\n"
            + _CLASSIFIER_INSTRUCTIONS
        )

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

    async def handle_message(self, msg: InboundMessage, preamble: str | None = None) -> None:
        logger.info(f"🔒 acquiring session lock for {msg.channel!r}")
        self._foreground_count += 1
        self._foreground_idle.clear()
        try:
            async with self._session.lock:
                await self._process(msg, preamble)
        finally:
            self._foreground_count -= 1
            if self._foreground_count == 0:
                self._foreground_idle.set()

    async def record_immediate(self, msg: InboundMessage, output: str) -> None:
        """Fast-path answer from the pre-agent classifier: deliver ``output`` to the
        channel and record the turn (user message + answer) into history WITHOUT
        invoking the main model, so the next full turn sees what was already done.

        Delivery happens first: if it fails, this raises before any history is written,
        letting the caller fall back to the full agent cleanly with no partial state."""
        if msg.channel != CRON_CHANNEL:
            await self._channel_manager.send_full_msg(msg.channel, output)
        async with self._session.lock:
            user_msg = ChatCompletionUserMessageParam(role="user", content=_persist_user_content(msg))
            assistant_msg = ChatCompletionAssistantMessageParam(role="assistant", content=output)
            self._session.history.append(user_msg)
            self._session.history.append(assistant_msg)
            self._append_to_checkpoint(user_msg)
            self._append_to_checkpoint(assistant_msg)

    async def handle_cron(self, job: CronJob) -> None:
        synthetic = InboundMessage(channel=CRON_CHANNEL, text=job.message, timestamp=0)
        self._foreground_count += 1
        self._foreground_idle.clear()
        try:
            async with self._session.lock:
                await self._process(synthetic, build_channel_ctx(CRON_CHANNEL))
        finally:
            self._foreground_count -= 1
            if self._foreground_count == 0:
                self._foreground_idle.set()

    async def _process(self, msg: InboundMessage, preamble: str | None = None) -> None:
        """Run one full agent turn: LLM call, tool loop, checkpoint, memory extraction."""
        self._tool_registry.current_channel = msg.channel
        try:
            await self._process_inner(msg, preamble)
        finally:
            self._tool_registry.current_channel = None

    async def _process_inner(self, msg: InboundMessage, preamble: str | None = None) -> None:
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

        # Ephemeral pre-message note built by the dispatch layer (main.on_message / handle_cron):
        # channel + datetime context, plus any fast-classifier error. Included in THIS turn's prompt
        # (system role isn't allowed mid-conversation, so it's a user note) but never persisted to
        # history — the datetime is regenerated every turn and old errors shouldn't linger.
        if preamble:
            messages.append(ChatCompletionUserMessageParam(role="user", content=preamble))

        # Images are written to disk here and referenced by marker; raises before any history is
        # touched, so a failed write leaves no partial state.
        user_msg = ChatCompletionUserMessageParam(role="user", content=_persist_user_content(msg))
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
        call_counts: dict[str, int] = {}  # identical tool calls this turn — see _REPEATED_CALL_LIMIT
        tool_counts: dict[str, int] = {}  # calls per tool NAME this turn — see _TOOL_CALLS_PER_TURN_LIMIT
        seen_interims: set[str] = set()  # interim lines already delivered this turn (dedup)

        while iterations < _MAX_TOOL_ITERATIONS:
            iterations += 1

            # Phase 1: tool-capable call (no response_model/response_format — lets model call tools freely)
            logger.info(f"🤖 LLM call start (iter={iterations}, msgs={len(messages)})")
            response = await self._llm.chat(_with_images(messages), tools=self._tool_registry.definitions)
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

            if response.tool_calls:
                # Build assistant message with tool_calls and append to history
                tool_content = response.content or ""
                if len(tool_content) > MAX_STORED_MSG_CHARS:
                    tool_content = tool_content[:MAX_STORED_MSG_CHARS] + "… [truncated]"
                assistant_msg = ChatCompletionAssistantMessageParam(role="assistant", content=tool_content)
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

                if iterations == _MAX_TOOL_ITERATIONS:
                    # Synthesize error responses so tool_calls are not orphaned in history.
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

                # Prose the model emits alongside tool calls is addressed to the user ("checking
                # which lights are on…"), so deliver it now — before the tools run, which is the
                # only moment it is still useful. Deliberately NOT setting turn_delivered: the
                # answer is still outstanding, so the safety net below must stay armed.
                interim = (response.content or "").strip()
                if interim and msg.channel != CRON_CHANNEL:
                    preview = interim[:120] + ("…" if len(interim) > 120 else "")
                    # Every interim is a real Signal message. A flailing turn repeats the same line
                    # on each tool call, which spammed a group with four identical messages — send
                    # each distinct line at most once per turn.
                    key = " ".join(interim.split()).casefold()
                    if key in seen_interims:
                        logger.info(f"🔇 suppressed duplicate interim to {msg.channel!r}: {preview!r}")
                    else:
                        seen_interims.add(key)
                        logger.info(f"📤 delivering interim to {msg.channel!r}: {preview!r}")
                        try:
                            await self._channel_manager.send_full_msg(msg.channel, interim)
                        except Exception as exc:
                            logger.warning(f"interim delivery to {msg.channel!r} skipped: {exc}")

                if msg.channel != CRON_CHANNEL:
                    try:
                        await self._channel_manager.send_chunk(msg.channel, "")
                        typing_signaled = True
                    except Exception as exc:
                        logger.warning(f"typing signal to {msg.channel!r} failed: {exc}")
                for tc in response.tool_calls:
                    def _trunc(v: object, n: int = 30) -> str:
                        s = json.dumps(v) if not isinstance(v, str) else v
                        return s if len(s) <= n else s[:n] + "..."
                    args_preview = "{" + ", ".join(f"{k}: {_trunc(v)}" for k, v in (tc.arguments or {}).items()) + "}"
                    logger.info(f"🔧 tool call: {tc.name!r} args={args_preview}")
                    sig = f"{tc.name}:{json.dumps(tc.arguments, sort_keys=True, default=str)}"
                    call_counts[sig] = call_counts.get(sig, 0) + 1
                    tool_counts[tc.name] = tool_counts.get(tc.name, 0) + 1
                    if tool_counts[tc.name] > _TOOL_CALLS_PER_TURN_LIMIT:
                        # The signature guard below is evaded by varying one argument: the observed
                        # loop wrote its reply to four differently-named files in a row.
                        logger.warning(f"🔁 blocked {tc.name!r} — {tool_counts[tc.name]} calls this turn")
                        result = (
                            f"error: you have called {tc.name} {tool_counts[tc.name] - 1} times this turn, "
                            "which is well past what this task needs. Changing an argument and calling it "
                            "again is not progress. Note your reply reaches the user through the `message` "
                            "field of your answer — it is delivered automatically, so it does not need to be "
                            "written to a file or sent with another tool. Write your answer now."
                        )
                    elif call_counts[sig] > _REPEATED_CALL_LIMIT:
                        # Observed live: 13 consecutive identical write_file calls creating a Python
                        # "runner" script, narrating "I'll run it" each time and never calling run_command.
                        # Repeating a call that already succeeded cannot make progress — say so.
                        logger.warning(f"🔁 blocked repeat #{call_counts[sig]} of {tc.name!r} — same arguments")
                        result = (
                            f"error: you already called {tc.name} with these exact arguments "
                            f"{call_counts[sig] - 1} time(s) this turn, so calling it again changes "
                            "nothing. Use a DIFFERENT tool for the next step. Note write_file only "
                            "saves a file, it never executes it — running something is run_command "
                            "(e.g. `python draw_square.py`). If the work is already done, stop "
                            "calling tools and write your answer."
                        )
                    else:
                        result = await self._tool_registry.execute(tc.name, tc.arguments)
                    messages.append(ChatCompletionToolMessageParam(role="tool", tool_call_id=tc.id, content=result))
                continue

            # No tool calls — Phase 2: structured call (response_format enforced, no real tools)
            if msg.channel == CRON_CHANNEL:
                break

            struct_msgs: list[ChatCompletionMessageParam] = list(messages)
            if response.content:
                struct_msgs.append(ChatCompletionAssistantMessageParam(role="assistant", content=response.content))
            struct_msgs.append(
                ChatCompletionUserMessageParam(
                    role="user",
                    content=(
                        f"The request you must answer NOW is: {_request_preview(msg.text)}\n"
                        "Anything asked earlier in this conversation is already done and superseded — "
                        "do not answer it again or resend its result.\n"
                        "Produce your Answer as a JSON object: justification (str), message (str), "
                        "attachments (list[str], omit when empty). message must never be empty — every "
                        "turn gets an answer. "
                        "IMPORTANT: if this task required a file edit or other tool action that has NOT yet been "
                        "performed in this turn, do NOT claim it in message — call the tool first. "
                        "If the user asked to be SENT a picture, put the filename from its "
                        "'[IMAGE <file> <mime>]' marker into attachments — that is the only way it is "
                        "delivered; describing or linking it does not send it."
                    ),
                )
            )
            struct_resp = await self._llm.chat(_with_images(struct_msgs), response_model=Answer)
            initial_answer: Answer | None = struct_resp.parsed if isinstance(struct_resp.parsed, Answer) else None
            if initial_answer is not None:
                raw_preview = initial_answer.message[:200] + ("…" if len(initial_answer.message) > 200 else "")
                logger.info(f"🗒️ agent answer (pre-review): msg={raw_preview!r}")

            if initial_answer is None:
                logger.warning(f"parse failure on {msg.channel!r} (iter={iterations}), retrying")
                if response.content:
                    messages.append(ChatCompletionAssistantMessageParam(role="assistant", content=response.content))
                messages.append(
                    ChatCompletionUserMessageParam(
                        role="user",
                        content=f"[{SYSERR} — your response did not parse as a valid Answer. "
                        "Retry with a valid JSON object: justification (str), message (str).]",
                    )
                )
                continue

            initial_content = initial_answer.message
            if len(initial_content) > MAX_STORED_MSG_CHARS:
                logger.warning(f"response truncated ({len(initial_content)} chars)")
                initial_content = initial_content[:MAX_STORED_MSG_CHARS] + "… [truncated]"
            assistant_msg = ChatCompletionAssistantMessageParam(role="assistant", content=initial_content)

            if not typing_signaled:
                try:
                    await self._channel_manager.send_chunk(msg.channel, "")
                    typing_signaled = True
                except Exception as exc:
                    logger.warning(f"typing signal to {msg.channel!r} failed: {exc}")

            att_markers: list[str] = []
            if not initial_answer.message.strip():
                review = Review(
                    is_correct=False,
                    to_be_fixed=[
                        "message is empty. Every turn must deliver an answer — write one. If you could not "
                        "do or find what was asked, say exactly that."
                    ],
                )
            else:
                # Resolve outbound attachments before review so the reviewer judges exactly what
                # would be sent. Names the model invented are dropped here and reported back to it.
                att_markers, att_rejected = _resolve_attachments(initial_answer.attachments)
                if att_rejected:
                    logger.warning(f"dropped unknown attachment(s): {att_rejected}")
                # The reviewer sees the pictures too. Without them it cannot verify a description of
                # an image, falls back on "the claimed tool call is missing", and rejects every
                # correct image answer — which then drives the agent into a doomed read_image hunt.
                review = await self._reviewer.run_review(
                    _with_images(messages),
                    assistant_msg,
                    initial_answer.justification,
                    attachment_parts=_expand_markers("\n".join(att_markers)) if att_markers else None,
                    current_request=msg.text,
                )
                if review.is_correct and att_rejected:
                    avail = _available_attachments(messages)
                    review = Review(
                        is_correct=False,
                        to_be_fixed=[
                            f"attachment(s) not sent: {'; '.join(att_rejected)}. "
                            + (
                                f"Copy one of these EXACTLY into attachments: {avail}."
                                if avail
                                else "No image has been fetched this turn — call download_image first, then "
                                "copy the filename from the '[IMAGE <file> <mime>]' marker it returns."
                            )
                        ],
                    )

            if review_start_idx < 0:
                review_start_idx = len(messages)
            messages.append(assistant_msg)

            if review.is_correct:
                review_accepted = True
                just_preview = initial_answer.justification[:200] + ("…" if len(initial_answer.justification) > 200 else "")
                logger.info(f"✅ review passed (attempt {review_rejections + 1}) justification={just_preview!r}")
                # An empty message is rejected before review, so an accepted answer always delivers.
                preview = initial_answer.message[:120] + ("…" if len(initial_answer.message) > 120 else "")
                att_note = f" +{len(att_markers)} attachment(s)" if att_markers else ""
                logger.info(f"📤 delivering to {msg.channel!r}: {preview!r}{att_note}")
                try:
                    await self._channel_manager.send_full_msg(
                        msg.channel, initial_answer.message, _load_attachments(att_markers)
                    )
                    turn_delivered = True
                except Exception as exc:
                    logger.warning(f"delivery to {msg.channel!r} skipped: {exc}")
                break

            review_rejections += 1
            reviewer_content = "[REVIEW — is_correct=False]"
            if review.to_be_fixed:
                reviewer_content += "\n" + "\n".join(f"- {p}" for p in review.to_be_fixed)
            # Only demand a tool call when the rejection actually asked for one. Appending it
            # unconditionally made the model re-run the same write_file+shell on every rejection,
            # burning all four retries without ever addressing what the reviewer objected to.
            fixes_text = " ".join(review.to_be_fixed).lower()
            tool_clause = ""
            if any(t in fixes_text for t in ("edit_file", "write_file", "run_command", "tool call", "tool result")):
                tool_clause = (
                    "The rejection cites a tool: you MUST call that tool in your very next response — "
                    "do NOT reword the claim without executing the action first. "
                )
            reviewer_content += (
                "\nRetry. " + tool_clause + "Address exactly what the rejection listed; do not redo work that "
                "already succeeded. Both fields required: justification (str), message (str), and message MUST be "
                "non-empty — the user is waiting for a response. Fix the specific problem and answer."
            )
            logger.info(f"❌ review rejected (attempt {review_rejections}/{MAX_NEGATIVE_REVIEWS}): {review.to_be_fixed}")
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
        if not self._ctx.should_compact(len(self._session.history)):
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
            # must expand images too, or the warmed prefix diverges from the real turn's
            await self._llm.chat(_with_images(messages), tools=self._tool_registry.definitions, max_tokens=1)
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
            # must match run_review's expansion, or the warmed prefix diverges
            await self._reviewer.warm_cache(_with_images(messages))
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
