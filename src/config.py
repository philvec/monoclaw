import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

logger = logging.getLogger("monoclaw")

SYSERR = "SYSTEM ERROR:"
CRON_CHANNEL = "cron"
WS_HOST = "0.0.0.0"
WS_PORT = 8765
MAX_STORED_MSG_CHARS = 8000
ARCHIVE_DIR = Path("./data/archive")
# Inbound images are written here and referenced from history by an "[IMAGE <file> <mime>]" marker;
# history itself stays str-only (see agent._persist_user_content). Deliberately outside
# ./data/workspace so the shell/write_file tools cannot reach them.
IMAGES_DIR = Path("./data/images")
# Explicit, because mimetypes.guess_extension("image/webp") is None on this Python: a fetched webp
# would be stored as "*.bin" and then rejected as an attachment for not looking like an image.
IMAGE_MIME_EXT = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
}


# What the LLM backend can actually decode: llama.cpp's mtmd uses stb_image (JPEG/PNG/GIF/BMP and
# some exotica) and has no WebP support at all — verified in the deployed build. Anything else must
# be converted BEFORE it is stored: an undecodable image in history makes llama-server 400 every
# later turn until it rolls out of the window, i.e. one bad picture bricks the assistant.
LLM_IMAGE_MIMES = {"image/jpeg", "image/png", "image/gif", "image/bmp"}


def to_decodable_image(data: bytes, mime: str) -> tuple[bytes, str]:
    """Return (data, mime) the LLM backend can read, converting if needed. Raises if impossible.

    A large share of web images are WebP, so converting beats refusing them.
    """
    if mime in LLM_IMAGE_MIMES:
        return data, mime
    import io

    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        buf = io.BytesIO()
        im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB").save(buf, format="PNG")
    return buf.getvalue(), "image/png"


def sniff_image_mime(head: bytes) -> str:
    """Identify an image from its magic bytes. Filenames are not trustworthy: a stored file may have
    no extension, or one the local mimetypes db has no entry for (notably .webp)."""
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if head.startswith(b"BM"):
        return "image/bmp"
    return ""


# Long-term memory: post-turn extraction, the pre-compaction flush, memory_search/memory_read and
# scheduled consolidation. OFF until Filip turns it back on. MASTER.md is NOT part of this — it is
# still injected into the system prompt and master_memory still works, because rule 5 needs it.
# Every prompt that names a memory tool is conditional on this flag: leaving the instructions in
# while the tools are gone would just teach the model to call something that is not there.
MEMORY_ENABLED = False

IMAGE_HISTORY_TURNS = 1  # newest USER marker messages shown as pictures; older lose their marker line
WS_MAX_FRAME_BYTES = 16 * 1024 * 1024  # base64 images exceed the websockets default (1 MB)
# Raw bytes of ONE outbound picture — each travels in its own message. base64 inflates by 4/3, and a
# message is one websocket frame: going over WS_MAX_FRAME_BYTES kills the channel mid-delivery.
OUTBOUND_IMAGES_MAX_TOTAL_BYTES = 8 * 1000 * 1000


class LLMConfig(BaseModel):
    base_url: str = "http://localhost:8080/v1"
    embeddings_url: str = ""  # separate embedding server; falls back to base_url if empty
    # Phase 1 still thinks, and reasoning is billed here while landing in reasoning_content — so at
    # 4096 it twice spent ~2min and returned "0 content chars, 0 tool call(s)", losing the tool call
    # it was about to make. Well inside the 65536 the server is started with. If truncation returns
    # at this budget the answer is no thinking on phase 1, not a bigger number: 8192 tokens of
    # reasoning for one edit_image call is looping, not thinking.
    max_tokens: int = 8192
    max_context: int = 32768  # practical context limit for compaction (0 = use model's reported window)
    max_history_messages: int = 100  # also triggers compaction when history exceeds this many messages
    compaction_keep_ratio: float = 0.25  # fraction of history to keep after compaction (rest is summarized)
    enable_thinking: bool = True


class ToolsConfig(BaseModel):
    exec_timeout_max_s: int = 600
    memory_ctx_trunc_n: int = 20
    memory_msg_max_len: int = 500
    memory_keep_recent: int = 10
    memory_decay_halflife_days: int = 30
    memory_embedding_weight: float = 0.6
    memory_mmr_lambda: float = 0.7
    memory_consolidation_cron: str = ""


class ClassifierConfig(BaseModel):
    # Fast pre-agent classifier (llama.cpp service, e.g. Qwen3.5-2B). Set via CLASSIFIER__BASE_URL,
    # analogous to LLM__BASE_URL for the main model. Empty ⇒ the whole layer is disabled at startup.
    base_url: str = ""
    # Whitelist of MCP tool names the classifier may call — same names/style as monoclaw-tools
    # TOOLS__ENABLED, over the SAME MCP servers the main model uses. Set via CLASSIFIER__TOOLS_ENABLED
    # (JSON list, e.g. '["ha_light_set"]'). NOTE the deliberate asymmetry: for the main model empty
    # means ALL tools, but for the classifier empty means NO tools — grant fast-path tools explicitly.
    tools_enabled: list[str] = []
    max_tokens: int = 512
    timeout_s: float = 8.0


class MCPServerConfig(BaseModel):
    name: str
    transport: Literal["stdio", "sse", "http"] = "stdio"
    command: str = ""
    args: list[str] = []
    env: dict[str, str] = {}
    url: str = ""


class Config(BaseSettings):
    llm: LLMConfig = LLMConfig()
    tools: ToolsConfig = ToolsConfig()
    classifier: ClassifierConfig = ClassifierConfig()
    mcp: list[MCPServerConfig] = []
    monoclaw_tools_url: str = ""  # shorthand: auto-registers monoclaw-tools sidecar when set

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        extra="ignore",
    )


def load_config(path: str = "config.yaml") -> Config:
    p = Path(path)
    data: dict[str, Any] = yaml.safe_load(p.read_text()) or {} if p.exists() else {}  # falls back to env vars only
    cfg = Config(**data)
    if cfg.monoclaw_tools_url:
        if any(s.name == "tools" for s in cfg.mcp):
            logger.warning(
                "MONOCLAW_TOOLS_URL is set but an MCP server named 'tools' already exists in config — not adding"
            )
        else:
            cfg.mcp.insert(0, MCPServerConfig(name="tools", transport="http", url=cfg.monoclaw_tools_url))
    return cfg
