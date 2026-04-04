import asyncio
import signal as _signal
import sys

from agent import AgentLoop
from channels import InboundMessage, WebSocketChannelManager
from config import load_config, logger
from context import ContextManager
from llm import LLMClient
from memory import MemoryManager
from scheduler import CronService
from tools import ToolRegistry


async def main() -> None:
    cfg = load_config()

    llm = LLMClient(cfg.llm)
    memory = MemoryManager(llm, cfg.tools)
    ctx = ContextManager(await llm.fetch_context_window(), cfg.llm.compaction_threshold)
    cron = CronService()
    channel_manager = WebSocketChannelManager()
    tool_registry = ToolRegistry.from_config(cfg, cron, channel_manager)

    agent = AgentLoop(llm, tool_registry, memory, ctx, channel_manager)

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
        loop.create_task(_cleanup(cron))

    for sig in (_signal.SIGINT, _signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown)
        except NotImplementedError:
            pass  # Windows

    logger.info("monoclaw starting")
    await channel_manager.start(on_message)
    await asyncio.Event().wait()  # block until shutdown signal


async def _cleanup(cron: CronService) -> None:
    await cron.stop()
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
