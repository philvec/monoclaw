from pydantic import BaseModel, Field


class Answer(BaseModel):
    justification: str = Field(
        description=(
            "Why you chose to reply or stay silent, AND — if replying — why you replied this way. "
            "Must cite the specific source that drove both the decision and the content: the exact "
            "tool result, the named memory entry, the specific past message, the exact channel rule, "
            "or a system prompt / MASTER.md rule (cite as 'system prompt states ...'). "
            "Explain why you relied on that source rather than another available one. "
            "Vague justifications ('seemed relevant', 'based on context') are not acceptable."
        ),
    )
    message: str = Field(
        default="",
        description="Exact text auto-delivered to the user on the INPUT CHANNEL. Empty when stay_silent=True.",
    )
    stay_silent: bool = Field(
        description=(
            "True = internal only, nothing is delivered (scratchpad / tool work). "
            "False = message is auto-delivered to the INPUT CHANNEL immediately. "
            "Apply channel rules from your system prompt when deciding."
        ),
    )


class Review(BaseModel):
    is_correct: bool = Field(
        description=(
            "True only if: the reply/silence decision is consistent with history and rules, "
            "the justification names a specific verifiable source, and that source genuinely "
            "supports both the decision and the message content."
        )
    )
    to_be_fixed: list[str] = Field(
        default_factory=list,
        description=(
            "Specific, actionable problems: inconsistencies, unverifiable facts, wrong decisions, "
            "or mismatches between the cited source and the message. Empty when is_correct=True."
        ),
    )
