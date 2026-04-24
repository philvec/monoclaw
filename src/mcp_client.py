import os
from contextlib import AsyncExitStack
from dataclasses import dataclass

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
import mcp.types as mcp_types

from config import MCPServerConfig, logger


@dataclass
class _MCPToolEntry:
    tool_name: str
    schema: dict
    session: ClientSession


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
        self._exit_stacks: list[AsyncExitStack] = []

    async def start(self) -> None:
        for cfg in self._configs:
            stack = AsyncExitStack()
            try:
                if cfg.transport == "stdio":
                    if not cfg.command:
                        logger.error(f"MCP server {cfg.name!r}: stdio transport requires 'command'")
                        await stack.aclose()
                        continue
                    params = StdioServerParameters(
                        command=cfg.command,
                        args=cfg.args,
                        env={**os.environ, **cfg.env} if cfg.env else None,
                    )
                    read, write = await stack.enter_async_context(stdio_client(params))
                elif cfg.transport == "sse":
                    if not cfg.url:
                        logger.error(f"MCP server {cfg.name!r}: sse transport requires 'url'")
                        await stack.aclose()
                        continue
                    read, write = await stack.enter_async_context(sse_client(cfg.url))
                else:  # http
                    if not cfg.url:
                        logger.error(f"MCP server {cfg.name!r}: http transport requires 'url'")
                        await stack.aclose()
                        continue
                    read, write, _ = await stack.enter_async_context(streamablehttp_client(cfg.url))

                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()

                result = await session.list_tools()
                registered = 0
                for tool in result.tools:
                    qname = f"{cfg.name}__{tool.name}"
                    if qname in self._tools:
                        logger.warning(f"MCP tool name collision: {qname!r} — skipping duplicate")
                        continue
                    self._tools[qname] = _MCPToolEntry(
                        tool_name=tool.name,
                        schema=_build_schema(qname, tool),
                        session=session,
                    )
                    registered += 1

                self._exit_stacks.append(stack)
                logger.info(f"MCP server {cfg.name!r}: connected, {registered} tools")
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                logger.error(f"MCP server {cfg.name!r}: failed to connect: {exc}")
                try:
                    await stack.aclose()
                except BaseException:
                    pass

    async def stop(self) -> None:
        for stack in reversed(self._exit_stacks):
            try:
                await stack.aclose()
            except Exception as exc:
                logger.warning(f"MCP shutdown error: {exc}")
        self._exit_stacks.clear()
        self._tools.clear()

    @property
    def tool_schemas(self) -> list[dict]:
        return [entry.schema for entry in self._tools.values()]

    async def execute(self, qualified_name: str, arguments: dict) -> str:
        entry = self._tools.get(qualified_name)
        if entry is None:
            return f"unknown MCP tool: {qualified_name!r}"
        try:
            result = await entry.session.call_tool(entry.tool_name, arguments=arguments)
            parts = []
            for c in result.content:
                if isinstance(c, mcp_types.TextContent):
                    parts.append(c.text)
                else:
                    parts.append(f"[{type(c).__name__}]")
            return "\n".join(parts) if parts else ""
        except Exception as exc:
            logger.error(f"MCP tool {qualified_name!r} raised: {exc}")
            return f"error: {exc}"
