"""
Fast pre-agent input classification layer, with optional tool actions.

Sits between the WebSocket transport (channels.py) and the Monoclaw agent
(agent.handle_message). Every inbound message is first shown to a small, fast,
local model (e.g. Qwen3.5-2B on the llama-cpp-classifier service) which returns
a constrained, structured verdict:

    language      = str (of the conversation)
    response_mode = "immediate" | "complex"
    output        = str
    tool_call     = { name, arguments } | null   (only when tools are configured)

- "complex":   ``output`` is sent at once as a "working on it" note (not recorded in
               history), then the message falls through to the full agent. Photos/videos
               are always complex.
- "immediate" + no tool_call: the layer answers the user directly with ``output``
               and records the turn into history, without calling the big model.
- "immediate" + tool_call: the layer executes the whitelisted MCP tool, delivers
               ``output`` as the confirmation, and records the turn. The big model
               is not called this turn.

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

import asyncio
import json
import random
import re
from pathlib import Path
from typing import Any, Literal

from openai import AsyncOpenAI
from pydantic import BaseModel

from channels import InboundMessage, WebSocketChannelManager
from config import CRON_CHANNEL, ClassifierConfig, logger

# Beside MASTER.md in the (gitignored) data volume; re-read per message for live edits.
SYSTEM_PROMPT_PATH = Path("./data/memory/fast_classifier_system.md")

# Qwen3.5's languages and dialects, one family per line, from the table in its release blog
# (qwen.ai/blog?id=qwen3.5). The `language` enum: free text gave "pl" and once "complex".
_LANGUAGES = [lang for family in (
    "English, French, Portuguese, German, Romanian, Swedish, Danish, Bulgarian, Russian, Czech, Greek, Ukrainian, "
    "Spanish, Dutch, Slovak, Croatian, Polish, Lithuanian, Norwegian Bokmål, Norwegian Nynorsk, Persian, Slovenian, "
    "Gujarati, Latvian, Italian, Occitan, Nepali, Marathi, Belarusian, Serbian, Luxembourgish, Venetian, Assamese, "
    "Welsh, Silesian, Asturian, Chhattisgarhi, Awadhi, Maithili, Bhojpuri, Sindhi, Irish, Faroese, Hindi, Punjabi, "
    "Bengali, Oriya, Tajik, Eastern Yiddish, Lombard, Ligurian, Sicilian, Friulian, Sardinian, Galician, Catalan, "
    "Icelandic, Tosk Albanian, Limburgish, Dari, Afrikaans, Macedonian, Sinhala, Urdu, Magahi, Bosnian, Armenian, "
    "Latgalian, Scottish Gaelic, Central Kurdish, Northern Kurdish, Southern Pashto, Sanskrit, Dhundari, Marwari, "
    "Ahirani, Bagheli, Bagri, Bundeli, Braj, Kumaoni, Kashmiri",
    "Simplified Chinese, Traditional Chinese, Cantonese, Burmese, Standard Tibetan, Meitei",
    "Arabic, Najdi Arabic, Levantine Arabic, Egyptian Arabic, Moroccan Arabic, Mesopotamian Arabic, "
    "Ta'izzi-Adeni Arabic, Tunisian Arabic, Gulf Arabic, Algerian Arabic, Sudanese Arabic, Libyan Arabic, Hebrew, "
    "Maltese, Amharic, Tigrinya, Kabyle, Somali, West Central Oromo, Hausa",
    "Indonesian, Malay, Tagalog, Cebuano, Javanese, Sundanese, Minangkabau, Balinese, Banjar, Pangasinan, Iloko, "
    "Waray (Philippines), Plateau Malagasy, Malagasy, Buginese, Maori, Samoan, Hawaiian, Fijian",
    "Tamil, Telugu, Kannada, Malayalam",
    "Turkish, North Azerbaijani, Northern Uzbek, Kazakh, Bashkir, Tatar, Crimean Tatar, Kyrgyz, Turkmen, Uyghur",
    "Thai, Lao, Shan",
    "Finnish, Estonian, Hungarian, Meadow Mari",
    "Vietnamese, Khmer",
    "Yoruba, Ewe, Kinyarwanda, Lingala, Northern Sotho, Nyanja, Shona, Southern Sotho, Tswana, Xhosa, Zulu, Luganda, "
    "Swati, Tsonga, Tumbuka, Venda, Chokwe, Luba-Kasai, Rundi, Umbundu, Kikuyu, Kongo, Nigerian Fulfulde, Wolof, Fon, "
    "Kabiyè, Mossi, Akan, Twi, Bambara, Igbo",
    "Japanese, Korean, Georgian, Basque, Haitian, Papiamento, Kabuverdianu, Tok Pisin, Swahili, Central Aymara, Tulu, "
    "Nagamese, Nigerian Pidgin, Mauritian Creole, Sango, Ayacucho Quechua, Halh Mongolian, Southwestern Dinka, Nuer, "
    "Guarani",
) for lang in family.split(", ")]

# Photo/video messages: complex only, no tool call — the classifier cannot see what it would act on.
_MEDIA_SCHEMA = {
    "type": "object",
    "properties": {
        "language": {"type": "string", "enum": _LANGUAGES},
        "response_mode": {"type": "string", "enum": ["complex"]},
        "output": {"type": "string", "minLength": 1},
    },
    "required": ["language", "response_mode", "output"],
    "additionalProperties": False,
}

# Confirmation of a completed tool action when the classifier's `output` came back empty. User-facing → Polish.
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

    language: str = ""
    response_mode: Literal["immediate", "complex"]
    output: str = ""
    tool_call: ToolCall | None = None


def _complete_tool_call(content: str) -> ToolCall | None:
    """The streamed verdict's tool_call once its JSON object is complete, if the verdict is immediate.
    language and response_mode are enums, so the first "tool_call" key is the real one."""
    if not re.search(r'"response_mode"\s*:\s*"immediate"', content) or (i := content.find('"tool_call"')) < 0:
        return None
    j = content.find(":", i) + 1
    while j < len(content) and content[j].isspace():
        j += 1
    try:
        value, _ = json.JSONDecoder().raw_decode(content, j)  # raises until the object has closed
    except json.JSONDecodeError:
        return None
    return ToolCall.model_validate(value) if isinstance(value, dict) else None


class Decision(BaseModel):
    handled: bool = False  # True ⇒ the layer answered/acted; do NOT call the agent this turn
    preamble: str | None = None  # non-None ⇒ inject this note before the agent turn (fail-safe)
    ack: str | None = None  # non-None ⇒ send this to the channel at once, before the agent turn


class FastClassifier:
    def __init__(self, cfg: ClassifierConfig, agent: object, mcp: object, channels: WebSocketChannelManager) -> None:
        self._cfg = cfg
        self._agent = agent
        self._mcp = mcp
        self._channels = channels
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
                "language": {"type": "string", "enum": _LANGUAGES},  # of the conversation: the ack's language
                "response_mode": {"type": "string", "enum": ["immediate", "complex"]},
                "output": {"type": "string", "minLength": 1},  # complex too: it is the instant ack
            },
            "required": ["language", "response_mode", "output"],
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
            # llama.cpp generates properties in this order: a tool confirmation follows the arguments it describes
            schema["properties"]["output"] = schema["properties"].pop("output")
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
        return "\n\n## Available tools (set tool_call when the request matches one)\n\n" + "\n\n".join(blocks)

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
        if not self.enabled or not (msg.text or msg.images) or msg.channel == CRON_CHANNEL:
            return Decision(handled=False)

        try:
            verdict, tool_task = await self._classify(msg)
        except Exception as exc:
            logger.error(f"⚡ fast classifier ERROR on {msg.channel!r}: {exc}")
            return Decision(handled=False, preamble=f"[FAST CLASSIFIER ERROR: {exc}]")

        if verdict.response_mode != "immediate":
            ack = verdict.output.strip()
            if not ack:
                logger.warning(f"⚡ COMPLEX with empty output on {msg.channel!r} — no ack sent")
                return Decision(handled=False)
            logger.info(
                f"⚡ classified COMPLEX [{msg.channel}] [{verdict.language}] — ack {ack!r}, passthrough to main agent"
            )
            msg.replies.append(ack)
            return Decision(handled=False, ack=ack)

        if verdict.tool_call is not None:
            return await self._run_tool(msg, verdict, tool_task)

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
        msg.replies.append(verdict.output)
        msg.answered = True
        return Decision(handled=True)

    async def _run_tool(
        self, msg: InboundMessage, verdict: FastClassification, tool_task: asyncio.Task | None
    ) -> Decision:
        tc = verdict.tool_call
        # The model's own wording ("OK, gaszę w sypialni"). tool_call precedes output in the schema, so it
        # is written after the arguments it describes.
        confirmation = verdict.output.strip()
        if not confirmation:
            confirmation = random.choice(_CONFIRM_TEMPLATES).format(subject=self._tool_summary(tc))
            logger.warning(f"⚡ tool {tc.name}: output empty on {msg.channel!r} — template confirmation")
        logger.info(
            f"⚡ classified IMMEDIATE/tool [{msg.channel}] tool={tc.name} args={tc.arguments} → {confirmation!r}"
        )
        # The tool usually started mid-stream (tool_task); a failed action is left to the main model to correct.
        sent, called = await asyncio.gather(
            self._channels.send_full_msg(msg.channel, confirmation),
            tool_task or self._mcp.call_checked(tc.name, tc.arguments),
            return_exceptions=True,
        )
        ok, result = (False, f"exception: {called}") if isinstance(called, BaseException) else called
        if not isinstance(sent, BaseException):
            msg.replies.append(confirmation)
        msg.replies.append(f"{tc.name}({json.dumps(tc.arguments, ensure_ascii=False)}) → {str(result)[:200]}")
        if not ok:
            logger.error(f"⚡ tool {tc.name} failed on {msg.channel!r}: {result}")
            if msg.channel.startswith("signal/"):
                try:
                    await self._channels.send_chunk(msg.channel, "")  # typing indicator until the agent's reply
                except Exception as exc:
                    logger.warning(f"typing signal to {msg.channel!r} failed: {exc}")
            told = "" if isinstance(sent, BaseException) else f" The user was already told: {confirmation!r}"
            return Decision(handled=False, preamble=f"[FAST CLASSIFIER ERROR: tool {tc.name} failed: {result}.{told}]")
        if isinstance(sent, BaseException):
            logger.error(f"⚡ confirmation delivery failed on {msg.channel!r}: {sent}")
            return Decision(handled=False, preamble=f"[FAST CLASSIFIER ERROR: confirmation delivery failed: {sent}]")
        logger.info(f"⚡ tool {tc.name} ok → {result!r}")
        msg.answered = True
        try:
            await self._agent.record_immediate(msg, confirmation, delivered=True)
        except Exception as exc:
            # Done and confirmed — handing it to the agent now would repeat the action.
            logger.error(f"⚡ recording the tool turn failed on {msg.channel!r}: {exc}")
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
        # No SDK retries: this call only runs on an already-failed turn, and the default two retries
        # would stretch a hung classifier to 3×timeout before the seed line ships.
        resp = await self._client.with_options(max_retries=0).chat.completions.create(
            model="local",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0.3,  # variety comes from the random draw of 3 lines; higher only garbles the grammar
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            timeout=self._cfg.timeout_s,
        )
        return (resp.choices[0].message.content or "").strip()

    async def _classify(self, msg: InboundMessage) -> tuple[FastClassification, asyncio.Task | None]:
        """Call the classifier model, streaming: a tool call starts the moment its JSON is complete, while
        ``output`` (its confirmation) is still being generated — returned as the running task. Raises on any
        failure before a tool started (caught by process())."""
        assert self._client is not None  # guaranteed while enabled

        system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()  # re-read for live edits
        if not system_prompt:
            raise ValueError(f"system prompt file is empty: {SYSTEM_PROMPT_PATH}")
        system_prompt += self._tools_doc  # append the self-documenting available-tools section
        # Compact one-line JSON: response_format enforces the SCHEMA, not conciseness — a pretty-printed nested verdict wastes ~34 decode tokens ≈ 0.8s/command (~1.9× slower). Verified 2026-07-20.
        system_prompt += (
            "\n\nOUTPUT FORMAT: return the verdict as compact single-line JSON, with no spaces, indentation "
            "or newlines (minified)."
        )
        # Without the replies it answers, a follow-up reads as a standalone message: "Dobrze zgadłeś!" was
        # echoed back, and "To jestem ja, Filip" (naming a person in a photo) got a fresh greeting. All of
        # them, tool calls included, even mid-turn: "Teraz zgaś" sent while a lamp was still being switched
        # on, shown only the last finished reply (a drawing's description), switched off "the drawing".
        # Never the previous message itself: the small model would answer that one instead.
        user = f"Message: {msg.text}"
        replies, answered = self._agent.previous_replies(msg)
        if replies:
            lines = "\n".join(f"- {r}" for r in replies)
            state = "" if answered else " (still in progress)"
            user = (
                f"[… previous conversation messages trimmed]\nResponses to the previous message{state}:\n{lines}"
                f"\n\n{user}"
            )
        schema = self._response_schema
        if msg.images:
            # The model runs without an mmproj: it would answer confidently about media it cannot see, so
            # the grammar allows only complex and the note asks for a "looking at it" ack.
            kind = "a video" if any(i.mime.startswith("video/") for i in msg.images) else "a photo"
            user += (
                f"\n[Attached: {kind}. You cannot see it; the full assistant will. In output, tell the user in "
                'the language of the conversation that you are looking at it (e.g. "Patrzę na zdjęcie...", '
                '"Oglądam filmik...", "Looking at the photo...").]'
            )
            schema = _MEDIA_SCHEMA

        content, early, tool_task = "", None, None
        try:
            stream = await self._client.chat.completions.create(
                model="local",  # llama.cpp ignores this field
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user},
                ],
                max_tokens=self._cfg.max_tokens,
                temperature=0.0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "FastClassification", "schema": schema},
                },
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},  # fast path, no reasoning
                timeout=self._cfg.timeout_s,
                stream=True,
            )
            async for chunk in stream:
                if not (chunk.choices and chunk.choices[0].delta.content):
                    continue
                content += chunk.choices[0].delta.content
                if tool_task is None and (early := _complete_tool_call(content)):
                    logger.info(f"⚡ tool {early.name} started mid-stream on {msg.channel!r}")
                    tool_task = asyncio.create_task(self._mcp.call_checked(early.name, early.arguments))
            if not content.strip():
                raise ValueError("classifier returned empty content")
            return FastClassification.model_validate_json(content), tool_task  # raises on invalid output
        except Exception as exc:
            if tool_task is None:
                raise
            # The action is already running: see it through, with the template confirmation.
            logger.error(f"⚡ classifier stream failed after {early.name} started on {msg.channel!r}: {exc}")
            return FastClassification(response_mode="immediate", tool_call=early), tool_task
