import json
from datetime import datetime, timezone
from typing import Any

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from config import ARCHIVE_DIR, logger, MEMORY_ENABLED
from llm import LLMClient
from models import Review

MAX_NEGATIVE_REVIEWS = 4
_MAX_REQUEST_CHARS = 2000  # group turns carry a whole transcript; keep the marker readable
# Reasoning is billed against max_tokens but returned in a separate field, so a long think block
# truncates the verdict itself: measured 836 completion tokens for a 20-char answer. A truncated
# verdict does not parse, and an unparsed review DEFAULTS TO APPROVED — which is how a turn that
# made no tool call at all shipped "engines added". The verdict's reasoning belongs in to_be_fixed.
_THINKING = False

# Gated on MEMORY_ENABLED for the same reason as the agent's own prompt: naming a tool that is not
# registered teaches the reviewer to demand it.
_MEM_SOURCE = "memory entry, " if MEMORY_ENABLED else ""
_MEM_NO_PENALTY = (
    " Do NOT penalise the agent for not calling `memory_search` when the answer is already there."
    if MEMORY_ENABLED
    else ""
)
_MEM_BACKED = " and NOT backed by a cited memory entry" if MEMORY_ENABLED else ""

# The output shape is grammar-enforced from the Review model (llm.chat sets response_format from it),
# and Review's own field descriptions state what to_be_fixed must contain — so neither is repeated here.
_REVIEW_PROMPT = f"""\
You are a reviewer verifying sense and compliance with the rules of a personal assistant agent's \
response.

# EVALUATION RULES

## 1. It must answer the current request
The given conversation history is context; the direct thing to respond to is the `[CURRENT REQUEST]` \
block. Earlier requests in the history are already handled — never reject a reply for not matching an \
older one, and never ask for an older result to be resent. A response answering a different question \
than the `[CURRENT REQUEST]` is `is_correct=False`. Admitting inability ("I could not find X", "that \
tool failed") IS a valid answer when that is in fact what happened.

## 2. The justification cites a specific, named source
An exact tool result, {_MEM_SOURCE}quoted past message, named channel rule, or system prompt / \
MASTER.md content — and it must verifiably support EVERY claim in the message. Facts stated directly \
in the agent system prompt (given as `[AGENT SYSTEM PROMPT]`, which includes injected MASTER.md \
rules) are pre-loaded and always available: no tool call is needed to cite them, and \
"system prompt rule: ..." or "MASTER.md states ..." is complete on its own.{_MEM_NO_PENALTY}

## 3. The cited source actually supports the claim
Do NOT accept message content at face value. "I searched and found nothing" requires the \
justification to cite the specific tool result that returned empty AND the rule directing the agent to \
inform the user of this. "I don't have access to X" requires it to cite the specific missing tool or \
data source. If the justification does not verifiably support a claim, mark `is_correct=False` \
regardless of how plausible or humble-sounding the message is.

## 4. Claimed tool calls really happened
`[ACTUALLY MADE TOOL CALLS THIS ROUND]` is the authoritative list of tools actually executed this \
turn; a tool absent from it was NOT called, whatever the history appears to show. **Tense does not \
matter** — if the message or justification says a tool was, is being, or is about to be used, verify \
its name appears in that list. A delivered response reports what was ACTUALLY DONE and never announces \
what is about to be done: the agent has a separate interim line for narration before a tool call, so a \
final answer saying "I am searching…" with no matching entry is fabricated. A justification written as \
intent ("I will first search, then write…") cites nothing and is not acceptable either. In such cases \
mark `is_correct=False` and return, to be corrected: "Call <tool> NOW, in this turn, and answer with \
its actual result — do not announce or promise an action you have not performed." so the agent knows \
what to improve.

## 5. External facts need a web search
If the message states a fact about a specific named entity (person, place, organisation, event), \
current data (price, schedule, ranking, availability), or any time-sensitive claim that is NOT present \
in the system prompt{_MEM_BACKED}, verify that `tools__web_search` appears in \
`[ACTUALLY MADE TOOL CALLS THIS ROUND]`. If the agent relied solely on training knowledge for such a \
claim, mark `is_correct=False`.

## 6. File modification claims need the tool that makes them
If the message claims a file was edited, written, modified, updated or created, verify that \
`edit_file` or `write_file` appears in `[ACTUALLY MADE TOOL CALLS THIS ROUND]`. If absent, mark \
`is_correct=False` and say: "Call edit_file (or write_file) in your next tool-use turn — do NOT claim \
the modification without a tool result confirming it; rewording the claim without calling the tool \
will be rejected again."

## 7. Pictures are attached for you too
Every picture in this conversation is attached to this review, so you see exactly what the assistant \
sees. Judge a description against the picture itself; "the picture shows ..." is a complete citation. \
Seeing a picture is not a tool call, so rule 4 does not reach it — rule 4 governs claims that a TOOL \
RAN, and nothing here weakens it. Reject only if the description contradicts the picture you see.

## 8. Pictures sent with this response
A `[PICTURES SENT THIS ROUND]` block holds the picture(s) already delivered to the user with this \
response — every one the assistant drew, changed or fetched this round, sent automatically as it was \
made. It cannot choose, name or add any other picture, so never ask it to. Mark `is_correct=False` if \
a picture does not show what the user asked for (wrong subject, wrong place, a logo/collage/screenshot \
instead of the thing itself, unreadable), or if the response describes it inaccurately; say what it \
actually shows. If the user asked to be sent a picture and there is no `[PICTURES SENT THIS ROUND]` \
block, nothing reached them: `is_correct=False`, and the only remedy is to call `generate_image` to \
draw one, or `image_search` then `download_image` to find a real one — never `send_message` or \
`send_email`, which reach the wrong place.

# TASK

Evaluate whether the assistant's response — the `[ASSISTANT RESPONSE TO REVIEW]` block — makes sense \
and is consistent with every rule above."""


class Reviewer:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    @staticmethod
    def _is_internal_scaffold(msg: ChatCompletionMessageParam) -> bool:
        """Return True for internal retry/error messages that should be hidden from the reviewer."""
        if msg.get("role") != "user":
            return False
        content = str(msg.get("content") or "")
        return content.startswith("[SYSTEM ERROR:") or content.startswith("[REVIEW — is_correct=False]")

    def _build_review_prefix(self, messages: list[ChatCompletionMessageParam]) -> list[ChatCompletionMessageParam]:
        """Build the stable reviewer prefix (everything except the response being reviewed)."""
        if messages and messages[0].get("role") == "system":
            agent_system_content = str(messages[0].get("content") or "")
            rest = messages[1:]
        else:
            agent_system_content = ""
            rest = messages

        review_msgs: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(role="system", content=_REVIEW_PROMPT),
        ]
        if agent_system_content:
            review_msgs.append(
                ChatCompletionUserMessageParam(role="user", content="[AGENT SYSTEM PROMPT]\n" + agent_system_content)
            )
        review_msgs.extend(m for m in rest if not self._is_internal_scaffold(m))
        return review_msgs

    async def warm_cache(self, messages: list[ChatCompletionMessageParam]) -> None:
        """Pre-warm the reviewer KV cache with the current history prefix."""
        review_msgs = self._build_review_prefix(messages)
        review_msgs.append(ChatCompletionUserMessageParam(role="user", content="."))
        await self._llm.chat(review_msgs, max_tokens=1, enable_thinking=_THINKING)

    async def run_review(
        self,
        messages: list[ChatCompletionMessageParam],
        assistant_msg: ChatCompletionAssistantMessageParam,
        justification: str = "",
        attachment_parts: list[Any] | None = None,
        current_request: str = "",
        called_tool_names: list[str] | None = None,
    ) -> Review:
        review_msgs = self._build_review_prefix(messages)
        assistant_content = str(assistant_msg.get("content") or "")
        # Without an explicit marker the reviewer has to guess which of the many requests in history
        # is the live one, and it reliably picks an earlier, better-established one: it rejected a
        # correct "two red circles on yellow" for not being the "originally requested" three circles
        # from two turns before, and the agent then sent the older image.
        ask = " ".join((current_request or "").split()) or "(not captured)"
        if len(ask) > _MAX_REQUEST_CHARS:
            ask = ask[:_MAX_REQUEST_CHARS] + "… [truncated]"
        # The justification lives only on the parsed Answer and is never part of `messages`. Without
        # it the reviewer — whose whole job is auditing it — sees nothing and rejects for "missing
        # justification" on a response that supplied one, which the agent then cannot fix.
        tool_list_str = "[" + ", ".join(called_tool_names) + "]" if called_tool_names else "(none)"
        review_msgs.append(
            ChatCompletionUserMessageParam(
                role="user",
                content=f"[CURRENT REQUEST — history above is context; this is what the response "
                f"must answer]\n{ask}\n\n"
                f"[ACTUALLY MADE TOOL CALLS THIS ROUND: {tool_list_str}]\n\n"
                f"[ASSISTANT JUSTIFICATION]\n{justification or '(none provided)'}\n\n"
                f"[ASSISTANT RESPONSE TO REVIEW]\n{assistant_content}",
            )
        )
        # Outbound pictures are injected explicitly rather than relied on from history: the reviewer
        # must see exactly what is about to be sent, whatever the history window happens to hold.
        if attachment_parts:
            review_msgs.append(
                ChatCompletionUserMessageParam(
                    role="user",
                    content=[
                        {
                            "type": "text",
                            "text": (
                                "[PICTURES SENT THIS ROUND] These image(s) have already been delivered "
                                "to the user with the response above. They follow, in order. Look at "
                                "them and judge whether they actually show what the user asked for and "
                                "match what the response claims."
                            ),
                        },
                        *attachment_parts,
                    ],
                )
            )
        resp = await self._llm.chat(review_msgs, response_model=Review, enable_thinking=_THINKING)
        if isinstance(resp.parsed, Review):
            return resp.parsed
        logger.warning(
            f"review parse failed (finish={resp.finish_reason}, error={resp.error!r}, msgs={len(review_msgs)}); "
            "defaulting is_correct=True"
        )
        return Review(is_correct=True, to_be_fixed=[])

    def archive_trail(self, trail: list[ChatCompletionMessageParam]) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        try:
            ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
            (ARCHIVE_DIR / f"review_{ts}.jsonl").write_text("\n".join(json.dumps(m) for m in trail) + "\n")
        except Exception as exc:
            logger.error(f"failed to archive review trail: {exc}")
