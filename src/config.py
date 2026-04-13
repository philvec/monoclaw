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


class LLMConfig(BaseModel):
    base_url: str = "http://localhost:8080/v1"
    embeddings_url: str = ""  # separate embedding server; falls back to base_url if empty
    max_tokens: int = 4096
    max_context: int = 32768  # practical context limit for compaction (0 = use model's reported window)
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
