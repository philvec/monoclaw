import json
from datetime import datetime, timezone
from typing import Any

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from config import ARCHIVE_DIR, logger
from llm import LLMClient
from models import Review

MAX_NEGATIVE_REVIEWS = 4
_MAX_REQUEST_CHARS = 2000  # group turns carry a whole transcript; keep the marker readable

_REVIEW_PROMPT = (
    "REVIEW (internal, not delivered). Evaluate the preceding assistant message: "
    "(1) Is the message non-empty and an actual answer to what was asked? What is being answered is "
    "the '[CURRENT REQUEST]' block below — judge against that one. "
    "The conversation above it is context, and you need it: it is what makes the current request "
    "intelligible (a request like 'send me the one with two' only means anything through it), it is "
    "where you verify tool calls and results, and it holds facts and earlier answers the reply may "
    "legitimately build on. Read it and use it for all of that. "
    "The one thing it is NOT is a menu of other requests to answer: the asks in it have already been "
    "handled. So never reject a response for failing to match what was asked 'originally' or in an "
    "earlier turn, and never tell the assistant to send a result that belongs to one of those. "
    "Every turn must deliver a reply — there is no silent option. An empty message, or one that "
    "answers a different question than the [CURRENT REQUEST], is is_correct=False. Admitting "
    "inability ('I could not find X', 'that tool failed') IS a valid answer when it is what happened. "
    "(2) Does the justification cite a specific, named source — exact tool result, memory entry, "
    "quoted past message, named channel rule, or system prompt / MASTER.md content — that verifiably "
    "supports EVERY claim in the message? "
    "IMPORTANT: facts stated directly in the agent system prompt (shown above as [AGENT SYSTEM PROMPT], "
    "which includes injected MASTER.md rules) are pre-loaded and always available — "
    "no tool call is required to cite them. Citing 'system prompt rule: ...' or 'MASTER.md states ...' "
    "is a valid and complete justification for any fact that actually appears there. "
    "Do NOT penalise the agent for not calling memory_search when the answer is already in the system prompt. "
    "(3) Does each cited source actually support what is claimed? "
    "Do NOT accept message content at face value. Every factual claim must be traceable to the justification: "
    "'I searched and found nothing' requires the justification to cite the specific tool result that returned empty "
    "AND the rule directing the agent to inform the user of this. "
    "'I don't have access to X' requires the justification to cite the specific missing tool or data source. "
    "If the justification does not verifiably support a claim, mark is_correct=False regardless of how "
    "plausible or humble-sounding the message is. "
    "(4) Tool call verification — TENSE DOES NOT MATTER. If the message or justification says a tool was, is "
    "being, or is about to be used, verify that the corresponding tool call (role=assistant with tool_calls) AND "
    "its result (role=tool) are actually present in THIS turn. This covers past ('I searched my memory', "
    "'I checked the web'), present ('I am searching', 'Szukam w sieci', 'Wyszukuję'), AND future/promissory "
    "('I will search', 'let me look', 'a potem napiszę', 'zaraz sprawdzę'). A delivered response must report what "
    "was ACTUALLY DONE, never announce what is about to be done: the agent gets a separate interim line for "
    "narration before a tool call, so a FINAL answer saying 'I am searching…' with no search in this turn is "
    "fabricated. Likewise a justification written as intent ('I will first search, then write…') cites nothing "
    "and is not acceptable. Mark is_correct=False and say in to_be_fixed: 'Call <tool> NOW, in this turn, and "
    "answer with its actual result — do not announce or promise an action you have not performed.' "
    "(5) Web search for external facts: if the message states a fact about a specific named entity "
    "(person, place, organisation, event), current data (price, schedule, ranking, availability), "
    "or any time-sensitive claim that is NOT present in the system prompt and NOT backed by a cited memory entry, "
    "verify that tools__web_search was called (role=assistant tool_calls containing web_search, "
    "followed by a role=tool result). If the agent relied solely on training knowledge for such a claim "
    "without searching, mark is_correct=False. "
    "(6) File modification claims: if the message claims a file was edited, written, modified, "
    "updated, or created, verify that edit_file or write_file tool calls (role=assistant with "
    "tool_calls) AND their results (role=tool) are present in THIS same turn. If those tool "
    "calls are absent, mark is_correct=False and include in to_be_fixed: 'Call edit_file (or "
    "write_file) in your next tool-use turn — do NOT claim the modification without a tool "
    "result confirming it; rewording the claim without calling the tool will be rejected again.' "
    "(7) Post-rejection evasion: if the conversation contains a [REVIEW — is_correct=False] message "
    "(meaning the agent's prior response was already rejected), the retry must actually address the "
    "problems cited. A retry that drops the content, answers something narrower, or hedges instead of "
    "fixing the cited problem is is_correct=False — say again, specifically, what still needs fixing. "
    "(8) Attached images: a message beginning with '[IMAGE <file> <mime>]' had that picture attached to it, "
    "and THE PICTURE IS ATTACHED TO THIS REVIEW TOO — you can see it. Judge the description against what you "
    "actually see in the image. The image itself is the source; 'the attached image shows ...' is a complete, "
    "acceptable citation. This is the ONE narrow exemption to check (4): looking at a picture that is ALREADY "
    "ATTACHED needs no tool call, because the agent sees it directly, exactly as you do — so do not demand "
    "read_image/read_file/run_command for it, and do not reject a description of it just because no tool was called. "
    "The exemption covers nothing else: it does NOT weaken check (4) for searches, file edits, or any other "
    "claimed, in-progress or promised action. The '[IMAGE ...]' filename is an internal reference, NOT a file in "
    "the workspace: do not ask anyone to open it. Reject only if the description contradicts the picture you see. "
    "(9) Outbound attachments: a '[CANDIDATE ATTACHMENTS]' message means the assistant is about to SEND "
    "those pictures to the user — look at them. Mark is_correct=False if a picture does not actually show "
    "what the user asked for (wrong subject, wrong place, a logo/collage/screenshot instead of the thing "
    "itself, unreadable), or if the response describes it inaccurately; say in to_be_fixed what the picture "
    "actually shows so the assistant can fetch a better one. Conversely, if the user asked to be SENT a "
    "picture and no candidate attachment is present, that is is_correct=False — a message containing only "
    "image URLs does NOT send a picture. In that case the ONLY correct remedy to give is: 'call download_image "
    "on one of the URLs, check the picture, then put that filename in the attachments field.' When choosing that "
    "remedy, do not tell the assistant to deliver via send_message or send_email to the person it is already "
    "replying to — its response and attachments auto-deliver to the input channel, so those are the wrong tools "
    "here and send_email in particular mails a real person by mistake. (This constrains which remedy you name; "
    "it never excuses a missing tool call under check (4).) "
    "Return ONLY a JSON object — no prose, no markdown, no explanation:\n"
    '{"is_correct": true, "to_be_fixed": []}\n'
    "Each entry in to_be_fixed must be a concrete, actionable problem."
)


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
        await self._llm.chat(review_msgs, max_tokens=1)

    async def run_review(
        self,
        messages: list[ChatCompletionMessageParam],
        assistant_msg: ChatCompletionAssistantMessageParam,
        justification: str = "",
        attachment_parts: list[Any] | None = None,
        current_request: str = "",
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
        review_msgs.append(
            ChatCompletionUserMessageParam(
                role="user",
                content=f"[CURRENT REQUEST — judge the response below against THIS. The conversation "
                f"above is background: use it to understand what this refers to and to verify tool "
                f"calls, but it is not an alternative request to answer]\n{ask}\n\n"
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
                                "[CANDIDATE ATTACHMENTS] The assistant wants to SEND the following "
                                "image(s) to the user with the response above. They follow, in order. "
                                "Look at them and judge whether they actually show what the user asked "
                                "for and match what the response claims."
                            ),
                        },
                        *attachment_parts,
                    ],
                )
            )
        resp = await self._llm.chat(review_msgs, response_model=Review)
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
