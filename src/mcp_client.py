import asyncio
import json
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
import mcp.types as mcp_types

from config import MCPServerConfig, logger

RECONNECT_DELAY = 5.0  # backoff between failed connection attempts
RECONNECT_TIMEOUT = 30.0  # how long a caller waits for its server to come back

_RESTART_MARKER = Path("./data/mcp_restart.json")
# Transport breakage a fresh process can plausibly clear. Deliberately narrow: tool-level
# failures (bad args, 404 from web_fetch) must never restart us.
_RESTART_HINTS = ("session terminated", "session not found", "server disconnected",
                  "connection refused", "connection reset", "all connection attempts failed")


def note_mcp_result(ok: bool, text: str, channel: str | None) -> None:
    """Last-resort escalation after the in-process reconnect above has already failed.

    Exits so Docker's ``restart: always`` rebuilds everything; main() then re-invokes the agent
    on ``channel``. The marker outlives the restart, so a second failure cannot crashloop — it
    blocks until a call succeeds, which is what a manual fix looks like from in here.
    """
    if ok:
        _RESTART_MARKER.unlink(missing_ok=True)  # tooling healthy again — re-arm
        return
    if _RESTART_MARKER.exists() or not any(h in text.lower() for h in _RESTART_HINTS):
        return
    _RESTART_MARKER.write_text(json.dumps({"channel": channel, "error": text[:200], "pending": True}))
    logger.error(f"MCP tooling unrecoverable ({text[:120]}) — restarting process")
    os._exit(1)


def consume_restart_marker() -> str | None:
    """Channel to resume after a self-restart, else None. Clears only ``pending``, so the marker
    keeps blocking further self-restarts until a tool call actually succeeds."""
    try:
        data = json.loads(_RESTART_MARKER.read_text())
    except (OSError, ValueError):
        return None
    if not data.get("pending"):
        return None
    data["pending"] = False
    _RESTART_MARKER.write_text(json.dumps(data))
    return data.get("channel")


@dataclass
class _MCPToolEntry:
    tool_name: str
    schema: dict
    session: ClientSession
    server: str


def _build_schema(qname: str, tool: mcp_types.Tool) -> dict:
    params = dict(tool.inputSchema)
    params.pop("title", None)
    return {
        "type": "function",
        "function": {
            "name": qname,
            "description": tool.description or "",
            "parameters": params,
        },
    }


class MCPClient:
    def __init__(self, server_configs: list[MCPServerConfig]) -> None:
        self._configs = server_configs
        self._tools: dict[str, _MCPToolEntry] = {}
        self._sessions: dict[str, ClientSession] = {}
        self._ready: dict[str, asyncio.Event] = {}
        self._reset_req: dict[str, asyncio.Event] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._tasks: list[asyncio.Task] = []
        self._closing = False

    async def start(self) -> None:
        for cfg in self._configs:
            self._ready[cfg.name] = asyncio.Event()
            self._reset_req[cfg.name] = asyncio.Event()
            self._locks[cfg.name] = asyncio.Lock()
            self._tasks.append(asyncio.create_task(self._serve(cfg), name=f"mcp-{cfg.name}"))
        # Block until every server has had one connect attempt (success or failure), so callers
        # see a fully-populated tool table — same guarantee the old inline start() gave.
        await asyncio.gather(*(self._ready[cfg.name].wait() for cfg in self._configs))

    async def stop(self) -> None:
        self._closing = True
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._tools.clear()
        self._sessions.clear()

    async def _serve(self, cfg: MCPServerConfig) -> None:
        """Own cfg's connection for its whole lifetime, rebuilding it on request.

        The connection must be opened *and* closed by this one task: the MCP transports wrap
        anyio task groups, and unwinding one from a different task raises "Attempted to exit
        cancel scope in a different task". Callers therefore never tear a session down
        themselves — they set the reset event and wait for this task to do it.
        """
        while not self._closing:
            self._ready[cfg.name].clear()
            self._reset_req[cfg.name].clear()
            try:
                async with AsyncExitStack() as stack:
                    session = await self._connect(cfg, stack)
                    count = self._register(cfg, session, (await session.list_tools()).tools)
                    logger.info(f"MCP server {cfg.name!r}: connected, {count} tools")
                    self._ready[cfg.name].set()
                    await self._reset_req[cfg.name].wait()
                    logger.warning(f"MCP server {cfg.name!r}: rebuilding connection")
            except Exception as exc:
                logger.error(f"MCP server {cfg.name!r}: failed to connect: {exc}")
                self._ready[cfg.name].set()  # never leave start()/callers blocked forever
                await asyncio.sleep(RECONNECT_DELAY)

    async def _connect(self, cfg: MCPServerConfig, stack: AsyncExitStack) -> ClientSession:
        if cfg.transport == "stdio":
            if not cfg.command:
                raise ValueError("stdio transport requires 'command'")
            params = StdioServerParameters(
                command=cfg.command,
                args=cfg.args,
                env={**os.environ, **cfg.env} if cfg.env else None,
            )
            read, write = await stack.enter_async_context(stdio_client(params))
        elif cfg.transport == "sse":
            if not cfg.url:
                raise ValueError("sse transport requires 'url'")
            read, write = await stack.enter_async_context(sse_client(cfg.url))
        else:  # http
            if not cfg.url:
                raise ValueError("http transport requires 'url'")
            read, write, _ = await stack.enter_async_context(streamablehttp_client(cfg.url))

        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    def _register(self, cfg: MCPServerConfig, session: ClientSession, tools) -> int:
        for qname in [q for q, e in self._tools.items() if e.server == cfg.name]:
            del self._tools[qname]
        self._sessions[cfg.name] = session
        registered = 0
        for tool in tools:
            qname = f"{cfg.name}__{tool.name}"
            if qname in self._tools:
                logger.warning(f"MCP tool name collision: {qname!r} — skipping duplicate")
                continue
            self._tools[qname] = _MCPToolEntry(
                tool_name=tool.name,
                schema=_build_schema(qname, tool),
                session=session,
                server=cfg.name,
            )
            registered += 1
        return registered

    async def _reset(self, server: str, stale: ClientSession) -> bool:
        """Have `server`'s owning task rebuild the connection. True once a newer session is live."""
        lock = self._locks.get(server)
        if lock is None:
            return False
        async with lock:
            if self._sessions.get(server) is not stale:
                return True  # a concurrent caller already rebuilt it
            self._ready[server].clear()
            self._reset_req[server].set()
            try:
                await asyncio.wait_for(self._ready[server].wait(), RECONNECT_TIMEOUT)
            except asyncio.TimeoutError:
                logger.error(f"MCP server {server!r}: reconnect timed out")
                return False
            return self._sessions.get(server) is not stale

    @property
    def tool_schemas(self) -> list[dict]:
        return [entry.schema for entry in self._tools.values()]

    def schemas_for(self, bare_names: list[str]) -> list[dict]:
        """Function schemas whose bare (unqualified) tool name is in ``bare_names`` — the same
        naming as monoclaw-tools TOOLS__ENABLED. Used to scope tools to the fast classifier."""
        return [entry.schema for entry in self._tools.values() if entry.tool_name in bare_names]

    async def _call(self, qualified_name: str, arguments: dict) -> tuple[bool, str]:
        """Invoke a tool, rebuilding the server connection once if the session has died."""
        for retry in (False, True):
            entry = self._tools.get(qualified_name)
            if entry is None:
                return False, f"unknown MCP tool: {qualified_name!r}"
            try:
                result = await entry.session.call_tool(entry.tool_name, arguments=arguments)
            except Exception as exc:
                # A dead session never heals itself: once the server restarts, our session id is
                # gone and every later call 404s with "Session terminated" — forever, because
                # nothing re-initializes. Rebuild the connection once, then retry the call.
                if retry or not await self._reset(entry.server, entry.session):
                    logger.error(f"MCP tool {qualified_name!r} raised: {exc}")
                    return False, f"error: {exc}"
                logger.warning(f"MCP tool {qualified_name!r} failed ({exc}) — reconnected, retrying")
                continue
            parts = [c.text if isinstance(c, mcp_types.TextContent) else f"[{type(c).__name__}]" for c in result.content]
            return (not getattr(result, "isError", False)), "\n".join(parts)
        return False, f"unknown MCP tool: {qualified_name!r}"  # unreachable

    async def call_checked(self, qualified_name: str, arguments: dict) -> tuple[bool, str]:
        """Invoke a tool, returning (ok, text) where ok is the inverse of the tool's isError flag —
        callers branch on it to decide between falling back and escalating."""
        return await self._call(qualified_name, arguments)
