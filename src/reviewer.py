import json
from datetime import datetime, timezone

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

_REVIEW_PROMPT = (
    "REVIEW (internal, not delivered). Evaluate the preceding assistant message: "
    "(1) Is the stay_silent decision explicitly and logically justified? "
    "If stay_silent=True, the justification MUST quote the exact rule or user instruction that permits silence "
    "(e.g. the specific channel rule from the system prompt, or the exact user message saying not to reply). "
    "A vague or implicit reason for silence is not acceptable — mark is_correct=False. "
    "(2) Does the justification cite a specific, named source — exact tool result, memory entry, "
    "quoted past message, named channel rule, or system prompt / MASTER.md content — that verifiably "
    "supports EVERY claim in the message and the reply decision? "
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
    "(4) Tool call verification: if the message or justification claims a tool was called or a search was performed "
    "(e.g. 'I searched my memory', 'I checked the web', 'I ran a search'), verify that the corresponding "
    "tool call (role=assistant with tool_calls) and its result (role=tool) are actually present in the "
    "conversation below. If those messages are absent, the claim is fabricated — mark is_correct=False. "
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
    ) -> Review:
        review_msgs = self._build_review_prefix(messages)
        assistant_content = str(assistant_msg.get("content") or "")
        review_msgs.append(
            ChatCompletionUserMessageParam(role="user", content="[ASSISTANT RESPONSE TO REVIEW]\n" + assistant_content)
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
