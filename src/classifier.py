"""
Fast pre-agent input classification layer, with optional tool actions.

Sits between the WebSocket transport (channels.py) and the Monoclaw agent
(agent.handle_message). Every inbound message is first shown to a small, fast,
local model (e.g. Qwen3.5-2B on the llama-cpp-classifier service) which returns
a constrained, structured verdict:

    response_mode = "immediate" | "complex"
    output        = str
    tool_call     = { name, arguments } | null   (only when tools are configured)

- "complex":   the layer logs the decision and lets the message fall through to
               the full agent, exactly as if the layer were not present.
- "immediate" + no tool_call: the layer answers the user directly with ``output``
               and records the turn into history, without calling the big model.
- "immediate" + tool_call: the layer executes the whitelisted MCP tool, delivers
               a short randomized confirmation, and records the turn. The big
               model is not called this turn.

B-hardened tool calls: the response schema is built dynamically from the MCP
tools' own arg schemas (see ``_build_response_schema``) as a discriminated union
(per-tool ``name`` enum + that tool's ``inputSchema``). Because llama.cpp grammar-
constrains generation to this schema, a tiny model cannot invent a tool name or
emit malformed arguments.

The system prompt lives beside MASTER.md at ``./data/memory/fast_classifier_system.md``
(gitignored, runtime-editable) and is re-read on every message.

Enable/disable is decided once at startup and logged once:
  - CLASSIFIER__BASE_URL unset          -> disabled (no classifier service)
  - fast_classifier_system.md missing   -> disabled (no system prompt)

Fail-safe: any runtime problem (llama error, invalid output, a failed/misfired
tool call, anything) is caught. The user's message still reaches the main agent,
prefixed with a "[FAST CLASSIFIER ERROR: <msg>]" note so monoclaw can react. The
layer can never drop, delay, or swallow a message.
"""

import random
from pathlib import Path
from typing import Any, Literal

from openai import AsyncOpenAI
from pydantic import BaseModel

from channels import InboundMessage
from config import CRON_CHANNEL, ClassifierConfig, logger

# Beside MASTER.md in the (gitignored) data volume; re-read per message for live edits.
SYSTEM_PROMPT_PATH = Path("./data/memory/fast_classifier_system.md")

# Confirmation phrasings for a completed tool action. The classifier's own `output` (a short human
# action subject) is logged but never delivered raw — it fills {subject}. User-facing → Polish.
_CONFIRM_TEMPLATES = [
    "Wykonano: {subject}.",
    "Zrobione! {subject}.",
    "Gotowe: {subject}.",
    "Ok, {subject}.",
    "Załatwione: {subject}.",
]

# abstention_line(): two one-line prompts — name the language, then paraphrase in it.
_LANG_PROMPT = "Name the language this message is written in. Output a single word."
_ABSTAIN_PROMPT = (
    "Formulate a single short sentence that paraphrases all three similar-meaning sentences below, "
    "in {lang} language — output only that sentence."
)
# Shuffled into the prompt above; also the verbatim reply if that model is unreachable.
_ABSTAIN_SEEDS = [
    "Sorry, I couldn't produce a coherent response for this one.",
    "I have to pass on this one — couldn't get to a verified answer.",
    "I'll have to sit this one out — couldn't verify my response.",
]


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = {}


class FastClassification(BaseModel):
    """Parsed classifier verdict. The response_format schema is built dynamically
    (``FastClassifier._build_response_schema``) so tool arguments are grammar-constrained
    per the MCP tool's own inputSchema; this model parses the result loosely."""

    response_mode: Literal["immediate", "complex"]
    output: str = ""
    tool_call: ToolCall | None = None


class Decision(BaseModel):
    handled: bool = False  # True ⇒ the layer answered/acted; do NOT call the agent this turn
    preamble: str | None = None  # non-None ⇒ inject this note before the agent turn (fail-safe)


class FastClassifier:
    def __init__(self, cfg: ClassifierConfig, agent: object, mcp: object) -> None:
        self._cfg = cfg
        self._agent = agent
        self._mcp = mcp
        self._client: AsyncOpenAI | None = None
        self._disabled_reason: str | None = None

        # Tools the classifier may call, selected from the SAME MCP servers the main model uses.
        # NOTE the deliberate asymmetry: for the main model an empty TOOLS__ENABLED means ALL tools,
        # but for the classifier an empty CLASSIFIER__TOOLS_ENABLED means NO tools — fast-path tools
        # must be granted explicitly.
        self._tool_schemas: list[dict] = mcp.schemas_for(cfg.tools_enabled) if cfg.tools_enabled else []
        self._response_schema = self._build_response_schema()
        self._tools_doc = self._build_tools_doc()  # self-documenting tools, appended to the prompt

        if not cfg.base_url:
            self._disabled_reason = "CLASSIFIER__BASE_URL not set — no classifier service address"
        elif not SYSTEM_PROMPT_PATH.exists() or not SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip():
            self._disabled_reason = f"system prompt file missing or empty: {SYSTEM_PROMPT_PATH}"
        else:
            self._client = AsyncOpenAI(base_url=cfg.base_url, api_key="sk-local")

    def _build_response_schema(self) -> dict:
        """Grammar-constrained response schema. With tools, ``tool_call`` is a discriminated union
        (per-tool ``name`` enum + that tool's own MCP arg schema); without tools it's omitted."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "response_mode": {"type": "string", "enum": ["immediate", "complex"]},
                "output": {"type": "string"},
            },
            "required": ["response_mode", "output"],
            "additionalProperties": False,
        }
        if self._tool_schemas:
            variants: list[dict] = [{"type": "null"}]
            for ts in self._tool_schemas:
                fn = ts["function"]
                variants.append(
                    {
                        "type": "object",
                        "description": fn.get("description", ""),
                        "properties": {
                            "name": {"type": "string", "enum": [fn["name"]]},
                            "arguments": fn.get("parameters", {"type": "object"}),
                        },
                        "required": ["name", "arguments"],
                        "additionalProperties": False,
                    }
                )
            schema["properties"]["tool_call"] = {"anyOf": variants}
            schema["required"].append("tool_call")
        return schema

    def _build_tools_doc(self) -> str:
        """Render the whitelisted tools (name, signature, description) into a prompt section, so the
        classifier learns each tool from the tool's OWN description — tools are self-documenting and
        the .md prompt stays generic (routing only). Empty when no tools are configured."""
        if not self._tool_schemas:
            return ""
        blocks = []
        for ts in self._tool_schemas:
            fn = ts["function"]
            sig = ", ".join(fn.get("parameters", {}).get("properties", {}).keys())
            blocks.append(f"### {fn['name']}({sig})\n{(fn.get('description') or '').strip()}")
        return "\n\n## Dostępne narzędzia (ustaw tool_call, gdy prośba do nich pasuje)\n\n" + "\n\n".join(blocks)

    @property
    def enabled(self) -> bool:
        return self._disabled_reason is None

    def log_startup(self) -> None:
        if self.enabled:
            tools = [ts["function"]["name"] for ts in self._tool_schemas]
            logger.info(
                f"⚡ fast classifier ENABLED — url={self._cfg.base_url}, prompt={SYSTEM_PROMPT_PATH}, tools={tools}"
            )
        else:
            logger.info(f"⚡ fast classifier DISABLED — {self._disabled_reason}")

    async def process(self, msg: InboundMessage) -> Decision:
        """Classify one inbound message and decide routing. Never raises."""
        if not self.enabled or not msg.text or msg.images or msg.channel == CRON_CHANNEL:
            if msg.images:
                # the classifier model runs without an mmproj: shown a caption but not the picture it
                # would answer confidently about an image it cannot see. Hand it to the agent instead.
                logger.info(f"⚡ image message on {msg.channel!r} — classifier has no vision, passthrough")
            return Decision(handled=False)

        try:
            verdict = await self._classify(msg)
        except Exception as exc:
            logger.error(f"⚡ fast classifier ERROR on {msg.channel!r}: {exc}")
            return Decision(handled=False, preamble=f"[FAST CLASSIFIER ERROR: {exc}]")

        if verdict.response_mode != "immediate":
            logger.info(f"⚡ classified COMPLEX [{msg.channel}] — passthrough to main agent")
            return Decision(handled=False)

        if verdict.tool_call is not None:
            return await self._run_tool(msg, verdict)

        # immediate, plain text answer
        preview = verdict.output[:120] + ("…" if len(verdict.output) > 120 else "")
        logger.info(f"⚡ classified IMMEDIATE/answer [{msg.channel}] → {preview!r}")
        if not verdict.output.strip():
            # An empty immediate answer delivers nothing but still records the turn as handled, so the
            # message would die here: no reply, no agent, no reviewer. Never a valid outcome.
            logger.warning(f"⚡ IMMEDIATE with empty output on {msg.channel!r} — passthrough to main agent")
            return Decision(handled=False, preamble="[FAST CLASSIFIER ERROR: immediate answer was empty]")
        try:
            await self._agent.record_immediate(msg, verdict.output)
        except Exception as exc:
            logger.error(f"⚡ immediate delivery failed on {msg.channel!r}: {exc}")
            return Decision(handled=False, preamble=f"[FAST CLASSIFIER ERROR: immediate delivery failed: {exc}]")
        return Decision(handled=True)

    async def _run_tool(self, msg: InboundMessage, verdict: FastClassification) -> Decision:
        tc = verdict.tool_call
        # Log the classifier's raw output + the tool call; the raw output is never delivered as-is.
        logger.info(
            f"⚡ classified IMMEDIATE/tool [{msg.channel}] tool={tc.name} args={tc.arguments} output={verdict.output!r}"
        )
        try:
            ok, result = await self._mcp.call_checked(tc.name, tc.arguments)
        except Exception as exc:
            ok, result = False, f"exception: {exc}"
        if not ok:
            # Fail-safe: the action did not succeed — hand the message to the main model with a note.
            logger.error(f"⚡ tool {tc.name} failed on {msg.channel!r}: {result}")
            return Decision(handled=False, preamble=f"[FAST CLASSIFIER ERROR: tool {tc.name} failed: {result}]")

        # Generic confirmation: tool name + ALL arguments (e.g. on/off), not the model's phrasing.
        subject = self._tool_summary(tc)
        confirmation = random.choice(_CONFIRM_TEMPLATES).format(subject=subject)
        logger.info(f"⚡ tool {tc.name} ok → {result!r}; reply {confirmation!r}")
        try:
            await self._agent.record_immediate(msg, confirmation)
        except Exception as exc:
            logger.error(f"⚡ confirmation delivery failed on {msg.channel!r}: {exc}")
            return Decision(handled=False, preamble=f"[FAST CLASSIFIER ERROR: confirmation delivery failed: {exc}]")
        return Decision(handled=True)

    @staticmethod
    def _tool_summary(tc: ToolCall) -> str:
        """Generic, tool-agnostic confirmation subject: tool name + all arguments (e.g. on/off)."""
        bare = tc.name.split("__", 1)[-1]
        args = ", ".join(f"{k}={v}" for k, v in tc.arguments.items())
        return f"{bare}({args})" if args else bare

    async def abstention_line(self, question: str) -> str:
        """The agent's "I can't answer this" line, worded by the small model in ``question``'s language.
        Never raises: the turn is already failing, so a seed line verbatim beats no reply at all."""
        try:
            lang = await self._ask(_LANG_PROMPT, question, 8)
            lines = "\n".join(random.sample(_ABSTAIN_SEEDS, len(_ABSTAIN_SEEDS)))
            if sentence := await self._ask(_ABSTAIN_PROMPT.format(lang=lang), lines, 100):
                logger.info(f"⚡ abstention line [{lang}] → {sentence!r}")
                return sentence
            logger.error("⚡ abstention line came back empty — falling back to a seed")
        except Exception as exc:
            logger.error(f"⚡ abstention line failed, falling back to a seed: {exc}")
        return random.choice(_ABSTAIN_SEEDS)

    async def _ask(self, system: str, user: str, max_tokens: int) -> str:
        assert self._client is not None  # guaranteed while enabled
        resp = await self._client.chat.completions.create(
            model="local",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0.3,  # variety comes from the random draw of 3 lines; higher only garbles the grammar
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            timeout=self._cfg.timeout_s,
        )
        return (resp.choices[0].message.content or "").strip()

    async def _classify(self, msg: InboundMessage) -> FastClassification:
        """Call the classifier model. Raises on any failure (caught by process())."""
        assert self._client is not None  # guaranteed while enabled

        system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()  # re-read for live edits
        if not system_prompt:
            raise ValueError(f"system prompt file is empty: {SYSTEM_PROMPT_PATH}")
        system_prompt += self._tools_doc  # append the self-documenting available-tools section
        # Compact one-line JSON: response_format enforces the SCHEMA, not conciseness — a pretty-printed nested verdict wastes ~34 decode tokens ≈ 0.8s/command (~1.9× slower). Verified 2026-07-20.
        system_prompt += "\n\nFORMAT WYJŚCIA: zwróć werdykt jako kompaktowy JSON w jednej linii, bez spacji, wcięć ani znaków nowej linii (minified)."

        resp = await self._client.chat.completions.create(
            model="local",  # llama.cpp ignores this field
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Kanał: {msg.channel}\nWiadomość: {msg.text}"},
            ],
            max_tokens=self._cfg.max_tokens,
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "FastClassification", "schema": self._response_schema},
            },
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},  # fast path, no reasoning
            timeout=self._cfg.timeout_s,
        )
        content = (resp.choices[0].message.content or "").strip() if resp.choices else ""
        if not content:
            raise ValueError("classifier returned empty content")
        return FastClassification.model_validate_json(content)  # raises on invalid / unparseable output
