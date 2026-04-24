import json
from datetime import datetime, timezone

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from config import ARCHIVE_DIR, MAX_STORED_MSG_CHARS, logger
from llm import LLMClient
from models import Answer, Review

MAX_NEGATIVE_REVIEWS = 3

_REVIEW_PROMPT = (
    "REVIEW (internal, not delivered). Evaluate the preceding assistant message: "
    "(1) Is the stay_silent decision explicitly and logically justified? "
    "If stay_silent=True, the justification MUST quote the exact rule or user instruction that permits silence "
    "(e.g. the specific channel rule from the system prompt, or the exact user message saying not to reply). "
    "A vague or implicit reason for silence is not acceptable — mark is_correct=False. "
    "(2) Does the justification cite a specific, named source — exact tool result, memory entry, "
    "quoted past message, or named channel rule — that verifiably supports EVERY claim in the message "
    "and the reply decision? "
    "(3) Does each cited source actually support what is claimed? "
    "Do NOT accept message content at face value. Every factual claim must be traceable to the justification: "
    "'I searched and found nothing' requires the justification to cite the specific tool result that returned empty "
    "AND the rule directing the agent to inform the user of this. "
    "'I don't have access to X' requires the justification to cite the specific missing tool or data source. "
    "If the justification does not verifiably support a claim, mark is_correct=False regardless of how "
    "plausible or humble-sounding the message is. "
    "Return ONLY a JSON object — no prose, no markdown, no explanation:\n"
    '{"is_correct": true, "to_be_fixed": []}\n'
    "Each entry in to_be_fixed must be a concrete, actionable problem."
)


class Reviewer:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def run_review_loop(
        self,
        messages: list[ChatCompletionMessageParam],
        initial_answer: Answer,
        initial_assistant_msg: ChatCompletionAssistantMessageParam,
    ) -> tuple[Answer | None, ChatCompletionAssistantMessageParam, list[ChatCompletionMessageParam], bool]:
        """
        Returns (final_answer, final_assistant_msg, archive_trail, max_reached).
        final_answer is None when max_reached=True — caller must not deliver.
        archive_trail contains every assistant attempt and every reviewer response in order.
        """
        answer = initial_answer
        assistant_msg = initial_assistant_msg
        archive_trail: list[ChatCompletionMessageParam] = [initial_assistant_msg]

        for attempt in range(MAX_NEGATIVE_REVIEWS):
            review = await self._run_review(messages, assistant_msg)
            reviewer_content = f"[REVIEW — is_correct={review.is_correct}]"
            if review.to_be_fixed:
                reviewer_content += "\n" + "\n".join(f"- {p}" for p in review.to_be_fixed)
            reviewer_msg = ChatCompletionUserMessageParam(role="user", content=reviewer_content)
            archive_trail.append(reviewer_msg)

            if review.is_correct:
                logger.info(f"review passed (attempt {attempt + 1})")
                return answer, assistant_msg, archive_trail, False

            logger.info(f"review rejected (attempt {attempt + 1}/{MAX_NEGATIVE_REVIEWS}): {review.to_be_fixed}")

            if attempt == MAX_NEGATIVE_REVIEWS - 1:
                logger.warning(f"max negative reviews ({MAX_NEGATIVE_REVIEWS}) reached, suppressing reply")
                return None, assistant_msg, archive_trail, True

            # Retry: agent sees only the last produced message + reviewer feedback, not accumulated attempts
            retry_msgs: list[ChatCompletionMessageParam] = [*messages, assistant_msg, reviewer_msg]
            retry_resp = await self._llm.chat(retry_msgs, response_model=Answer)
            if isinstance(retry_resp.parsed, Answer):
                new_answer = retry_resp.parsed
            else:
                logger.warning(f"retry parse failed (attempt {attempt + 1}); staying silent")
                new_answer = Answer(justification="parse failed on retry", message="", stay_silent=True)
            new_content = "[STAYED SILENT]" if new_answer.stay_silent else new_answer.message
            if len(new_content) > MAX_STORED_MSG_CHARS:
                logger.warning(f"retry response truncated ({len(new_content)} chars, attempt {attempt + 1})")
                new_content = new_content[:MAX_STORED_MSG_CHARS] + "… [truncated]"
            new_assistant_msg = ChatCompletionAssistantMessageParam(role="assistant", content=new_content)
            archive_trail.append(new_assistant_msg)
            answer = new_answer
            assistant_msg = new_assistant_msg

        return None, assistant_msg, archive_trail, True  # unreachable; satisfies type checker

    async def _run_review(
        self,
        messages: list[ChatCompletionMessageParam],
        assistant_msg: ChatCompletionAssistantMessageParam,
    ) -> Review:
        # Reviewer gets its own system prompt. The agent's system prompt is demoted to a user
        # message so it stays visible as context without overriding reviewer instructions.
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
        review_msgs.extend(rest)
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
