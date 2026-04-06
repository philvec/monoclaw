import logging
from pathlib import Path
from typing import Any

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
    compaction_threshold: float = 0.85
    enable_thinking: bool = True


class ToolsConfig(BaseModel):
    brave_api_key: str = ""
    exec_timeout_max_s: int = 600
    memory_ctx_trunc_n: int = 20
    memory_msg_max_len: int = 500
    memory_keep_recent: int = 10
    memory_decay_halflife_days: int = 30
    memory_embedding_weight: float = 0.6
    memory_mmr_lambda: float = 0.7
    memory_consolidation_cron: str = ""


class Config(BaseSettings):
    llm: LLMConfig = LLMConfig()
    tools: ToolsConfig = ToolsConfig()

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        extra="ignore",
    )


def load_config(path: str = "config.yaml") -> Config:
    p = Path(path)
    data: dict[str, Any] = yaml.safe_load(p.read_text()) or {} if p.exists() else {}  # falls back to env vars only
    return Config(**data)
