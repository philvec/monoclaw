import asyncio
import signal as _signal
import sys
import time
from pathlib import Path

from agent import AgentLoop, build_channel_ctx
from channels import InboundMessage, WebSocketChannelManager
from classifier import FastClassifier
from config import load_config, logger, MEMORY_ENABLED
from context import ContextManager
from llm import LLMClient
from mcp_client import MCPClient, consume_restart_marker
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
        max_history_messages=cfg.llm.max_history_messages,
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
    if MEMORY_ENABLED and cfg.tools.memory_consolidation_cron:
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

    # Pre-agent input classification layer: sits right after the WebSocket
    # (channels.py) and before the Monoclaw agent. Decides per message whether to
    # answer immediately (fast path) or pass through to the full agent.
    fast_classifier = FastClassifier(cfg.classifier, agent, mcp)
    fast_classifier.log_startup()
    agent.attach_abstention_line(fast_classifier.abstention_line)  # same small model words the abstention

    async def on_message(msg: InboundMessage) -> None:
        try:
            decision = await fast_classifier.process(msg)
        except Exception:
            # Defensive: process() is already fail-safe, but never let the layer block a message.
            logger.exception(f"fast classifier layer crashed on {msg.channel!r}; falling through to agent")
            decision = None
        if decision is not None and decision.handled:
            return  # immediate: the layer answered and recorded the turn

        # Build the per-turn context (channel + datetime) here, at the dispatch layer, and prepend
        # any fast-classifier error note. Both travel to the agent via the ephemeral `preamble`.
        ctx = build_channel_ctx(msg.channel)
        error_note = decision.preamble if decision is not None else None
        preamble = f"{error_note}\n{ctx}" if error_note else ctx
        try:
            await agent.handle_message(msg, preamble=preamble)
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

    # Came back from a self-restart (mcp_client) — tell the agent tooling is usable again.
    if (resume_channel := consume_restart_marker()) is not None:
        async def _resume() -> None:
            await asyncio.sleep(5)  # let channel clients (signal-bridge) reconnect first
            await agent.handle_message(InboundMessage(
                channel=resume_channel, text="solved, you can now use the tool retry",
                timestamp=int(time.time())))
        asyncio.create_task(_resume(), name="mcp-resume")

    logger.info("monoclaw starting")
    await channel_manager.start(on_message)


async def _cleanup(cron: CronService, mcp: MCPClient) -> None:
    await cron.stop()
    await mcp.stop()
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
