from pydantic import BaseModel, Field


class Answer(BaseModel):
    justification: str = Field(
        description=(
            "Why you replied this way. Must cite the specific source that drove the content: the exact "
            "tool result, the named memory entry, the specific past message, the exact channel rule, "
            "or a system prompt / MASTER.md rule (cite as 'system prompt states ...'). "
            "Explain why you relied on that source rather than another available one. "
            "Vague justifications ('seemed relevant', 'based on context') are not acceptable."
        ),
    )
    message: str = Field(
        description=(
            "Exact text auto-delivered to the user on the INPUT CHANNEL. Must never be empty — every "
            "turn gets an answer, even if that answer is that you could not do or find something."
        ),
    )


class Review(BaseModel):
    is_correct: bool = Field(
        description=(
            "True only if: the reply is consistent with history and rules, the justification names "
            "a specific verifiable source, and that source genuinely supports the message content."
        )
    )
    to_be_fixed: list[str] = Field(
        default_factory=list,
        description=(
            "Specific, actionable problems: inconsistencies, unverifiable facts, wrong decisions, "
            "or mismatches between the cited source and the message. Empty when is_correct=True."
        ),
    )
