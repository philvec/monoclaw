import asyncio
import signal as _signal
import sys
from pathlib import Path

from agent import AgentLoop
from channels import InboundMessage, WebSocketChannelManager
from config import load_config, logger
from context import ContextManager
from llm import LLMClient
from mcp_client import MCPClient
from memory import MemoryManager
from memory_store import MemoryStore
from scheduler import CronService, CronSchedule
from tools import ToolRegistry


async def main() -> None:
    cfg = load_config()

    llm = LLMClient(cfg.llm)

    store = MemoryStore(
        Path("./data/memory"),
        halflife_days=cfg.tools.memory_decay_halflife_days,
        embedding_weight=cfg.tools.memory_embedding_weight,
        mmr_lambda=cfg.tools.memory_mmr_lambda,
    )
    memory = MemoryManager(llm, cfg.tools, store)

    model_ctx = await llm.fetch_context_window()
    context_limit = cfg.llm.max_context if cfg.llm.max_context > 0 else model_ctx
    logger.info(f"context limit: {context_limit} (model reports {model_ctx})")
    ctx = ContextManager(
        context_limit,
        keep_recent=cfg.tools.memory_keep_recent,
        keep_ratio=cfg.llm.compaction_keep_ratio,
    )
    cron = CronService()
    channel_manager = WebSocketChannelManager()
    tool_registry = ToolRegistry.from_config(cfg, cron, channel_manager, store, llm)

    mcp = MCPClient(cfg.mcp)
    await mcp.start()
    tool_registry.attach_mcp(mcp)

    agent = AgentLoop(llm, tool_registry, memory, ctx, channel_manager)
    await agent.startup()

    # optional: scheduled memory consolidation
    if cfg.tools.memory_consolidation_cron:
        try:
            cron.add_job(
                schedule=CronSchedule(type="cron", cron_expr=cfg.tools.memory_consolidation_cron),
                message=(
                    "[SYSTEM] Run memory consolidation: use memory_search to review all memories. "
                    "Merge duplicates, update stale entries, delete irrelevant ones."
                ),
                name="memory-consolidation",
            )
            logger.info(f"memory consolidation scheduled: {cfg.tools.memory_consolidation_cron}")
        except Exception as exc:
            logger.error(f"failed to schedule memory consolidation: {exc}")

    async def on_message(msg: InboundMessage) -> None:
        logger.info(f"message from {msg.channel!r}: {msg.text[:50]!r}")
        try:
            await agent.handle_message(msg)
        except Exception:
            logger.exception(f"unhandled error processing message from {msg.channel!r}")

    await cron.start(on_trigger=agent.handle_cron)

    loop = asyncio.get_running_loop()

    def _shutdown() -> None:
        logger.info("shutdown signal received")
        loop.create_task(_cleanup(cron, mcp))

    for sig in (_signal.SIGINT, _signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown)
        except NotImplementedError:
            pass  # Windows

    logger.info("monoclaw starting")
    await channel_manager.start(on_message)


async def _cleanup(cron: CronService, mcp: MCPClient) -> None:
    await cron.stop()
    await mcp.stop()
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
