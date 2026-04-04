import asyncio
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar, cast

import httpx
from markdownify import markdownify
from pydantic import BaseModel, Field

from channels import WebSocketChannelManager
from config import Config, logger

from datetime import datetime, timezone
from scheduler import CronSchedule, CronService

_WORKSPACE = Path("./data/workspace")

P = TypeVar("P", bound=BaseModel)


class Tool(ABC, Generic[P]):
    """Abstract base for all agent tools.

    To add a new tool, subclass Tool and define:
    - A docstring — becomes the tool description shown to the LLM.
    - An inner ``Params(BaseModel)`` class — fields become the JSON Schema parameters.
    - An ``async def execute(self, params: Params)`` method.

    ``name`` is derived automatically from the class name (``FooBarTool`` → ``foo_bar``).
    Override the ``name`` property if the default is unsuitable.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

    @property
    def name(self) -> str:
        return re.sub(r"(?<!^)(?=[A-Z])", "_", type(self).__name__.removesuffix("Tool")).lower()

    @property
    def description(self) -> str:
        return (type(self).__doc__ or "").strip()

    class Params(BaseModel):
        pass

    @abstractmethod
    async def execute(self, params: P) -> Any: ...

    def to_schema(self) -> dict:
        schema = self.Params.model_json_schema()
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }


class ToolRegistry:
    """Registry of available tools; dispatches execution by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool[Any]] = {}

    @classmethod
    def from_config(cls, cfg: Config, cron: CronService, channel_manager: WebSocketChannelManager) -> "ToolRegistry":
        registry = cls()
        for tool in cast(
            list[Tool[Any]],
            [
                ReadFileTool(cfg),
                WriteFileTool(cfg),
                EditFileTool(cfg),
                GlobTool(cfg),
                GrepTool(cfg),
                ShellTool(cfg),
                WebSearchTool(cfg),
                WebFetchTool(cfg),
                ScheduleTool(cfg, cron),
                SetOutputChannelTool(cfg, channel_manager),
                ListOutputChannelsTool(cfg, channel_manager),
            ],
        ):
            registry.register(tool)
        return registry

    def register(self, tool: Tool[Any]) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool[Any]:
        return self._tools[name]

    @property
    def definitions(self) -> list[dict]:
        return [t.to_schema() for t in self._tools.values()]

    async def execute(self, name: str, arguments: dict) -> str:
        if name not in self._tools:
            return f"unknown tool: {name!r}"
        tool = self._tools[name]
        try:
            result = await tool.execute(tool.Params.model_validate(arguments))
        except Exception as exc:
            logger.exception(f"tool {name!r} raised")
            return f"error: {exc}"
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result)
        except (TypeError, ValueError):
            return str(result)


def _safe_path(rel: str) -> Path:
    """Resolve rel relative to _WORKSPACE; raise ValueError if it escapes."""
    base = _WORKSPACE.resolve()
    target = (base / rel).resolve()
    if not str(target).startswith(str(base)):
        raise ValueError(f"path {rel!r} escapes workspace")
    return target


class ReadFileTool(Tool["ReadFileTool.Params"]):
    """Read the contents of a file inside the workspace."""

    class Params(BaseModel):
        path: str = Field(description="Relative path to file")
        offset: int = Field(default=1, description="Line to start reading from (1-indexed)")
        limit: int = Field(default=500, description="Max lines to read")

    async def execute(self, params: Params) -> str:  # type: ignore[override]
        target = _safe_path(params.path)
        if not target.exists():
            return f"file not found: {params.path}"
        lines = target.read_text(errors="replace").splitlines()
        start = max(0, params.offset - 1)
        chunk = lines[start : start + params.limit]
        return "\n".join(f"{start + i + 1}\t{line}" for i, line in enumerate(chunk))


class WriteFileTool(Tool["WriteFileTool.Params"]):
    """Write content to a file inside the workspace, creating it if needed."""

    class Params(BaseModel):
        path: str
        content: str

    async def execute(self, params: Params) -> str:  # type: ignore[override]
        target = _safe_path(params.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(params.content)
        return f"wrote {len(params.content)} chars to {params.path}"


class EditFileTool(Tool["EditFileTool.Params"]):
    """Replace an exact string in a file."""

    class Params(BaseModel):
        path: str
        old_string: str = Field(description="Exact text to find")
        new_string: str = Field(description="Replacement text")

    async def execute(self, params: Params) -> str:  # type: ignore[override]
        target = _safe_path(params.path)
        if not target.exists():
            return f"file not found: {params.path}"
        text = target.read_text(errors="replace")
        if params.old_string not in text:
            return f"old_string not found in {params.path}"
        target.write_text(text.replace(params.old_string, params.new_string, 1))
        return f"edited {params.path}"


class GlobTool(Tool["GlobTool.Params"]):
    """List files matching a glob pattern inside the workspace."""

    class Params(BaseModel):
        pattern: str = Field(description="Glob pattern, e.g. '**/*.py'")

    async def execute(self, params: Params) -> str:  # type: ignore[override]
        base = _WORKSPACE.resolve()
        matches = sorted(str(p.relative_to(base)) for p in base.glob(params.pattern))
        if not matches:
            return "no matches"
        return "\n".join(matches)


class GrepTool(Tool["GrepTool.Params"]):
    """Search file contents for a regex pattern."""

    class Params(BaseModel):
        pattern: str = Field(description="Regex pattern")
        path: str = Field(default=".", description="File or directory to search (relative)")
        glob: str = Field(default="*", description="File glob filter, e.g. '*.py'")

    async def execute(self, params: Params) -> str:  # type: ignore[override]
        base = _WORKSPACE.resolve()
        target = _safe_path(params.path)

        try:
            regex = re.compile(params.pattern)
        except re.error as exc:
            return f"invalid regex: {exc}"

        results: list[str] = []
        files = [target] if target.is_file() else target.rglob(params.glob)
        for f in files:
            if not f.is_file():
                continue
            try:
                for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                    if regex.search(line):
                        rel = f.relative_to(base)
                        results.append(f"{rel}:{i}: {line}")
            except OSError:
                continue
            if len(results) >= 200:
                break

        return "\n".join(results) if results else "no matches"


class ShellTool(Tool["ShellTool.Params"]):
    """Execute a shell command. Working directory is the workspace."""

    _DENY_PATTERNS: list[re.Pattern[str]] = [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"\brm\s+-[a-z]*r[a-z]*f",  # rm -rf / rm -fr
            r"\bdd\b.*\bof=",  # dd of=...
            r"\bmkfs\b",  # mkfs.*
            r"\breboot\b",  # reboot
            r"\bshutdown\b",  # shutdown
            r"\bhalt\b",  # halt
            r"\bpoweroff\b",  # poweroff
            r"\bchmod\s+777\b",  # chmod 777
            r"\bchmod\s+-R\b",  # chmod -R (overly broad)
            r"curl[^|]+\|\s*(ba)?sh",  # curl ... | bash/sh
            r"wget[^|]+\|\s*(ba)?sh",  # wget ... | bash/sh
            r">\s*/dev/(s?d[a-z]|nvme)",  # writing to block devices
            r"\.\./",  # path traversal
            r"\.\.[/\\]",  # path traversal (Windows style)
            r":\s*\(\s*\)\s*\{.*:[^}]*\}",  # fork bomb: :(){ :|:& };
        ]
    ]

    class Params(BaseModel):
        command: str
        timeout: int = Field(default=60, description="Max seconds to wait")

    @staticmethod
    def _classify(command: str) -> str | None:
        """Return a denial reason if the command is blocked, else None."""
        for pat in ShellTool._DENY_PATTERNS:
            if pat.search(command):
                return f"command blocked by safety classifier (matched: {pat.pattern!r})"
        return None

    async def execute(self, params: Params) -> str:  # type: ignore[override]
        denial = self._classify(params.command)
        if denial:
            return f"blocked: {denial}"
        timeout = min(params.timeout, self._cfg.tools.exec_timeout_max_s)
        try:
            proc = await asyncio.create_subprocess_shell(
                params.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(_WORKSPACE.resolve()),
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            exit_code, output = proc.returncode or 0, stdout.decode(errors="replace")
        except asyncio.TimeoutError:
            exit_code, output = 124, "timed out"
        except Exception as exc:
            exit_code, output = 1, str(exc)
        suffix = f"\n[exit {exit_code}]" if exit_code != 0 else ""
        return _truncate(output) + suffix


def _truncate(text: str, max_chars: int = 10_000) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f"\n...[truncated {len(text) - max_chars} chars]...\n" + text[-half:]


class WebSearchTool(Tool["WebSearchTool.Params"]):
    """Search the web and return a summary of results."""

    class Params(BaseModel):
        query: str
        count: int = Field(default=5, description="Number of results")

    async def execute(self, params: Params) -> str:  # type: ignore[override]
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": params.query, "count": params.count},
                    headers={"Accept": "application/json", "X-Subscription-Token": self._cfg.tools.brave_api_key},
                    timeout=10.0,
                )
                resp.raise_for_status()
                results = resp.json().get("web", {}).get("results", [])
                return self._format_results(results)
        except Exception as exc:
            return f"search error: {exc}"

    @staticmethod
    def _format_results(results: list[dict]) -> str:
        if not results:
            return "no results"
        lines = []
        for r in results:
            lines.append(f"[{r.get('title', '')}]({r.get('url', '')})")
            snippet = r.get("description", "")
            if snippet:
                lines.append(snippet)
            lines.append("")
        return "\n".join(lines).strip()


class WebFetchTool(Tool["WebFetchTool.Params"]):
    """Fetch a URL and return its content as markdown."""

    class Params(BaseModel):
        url: str
        max_chars: int = Field(default=20_000, description="Max chars to return (default 20000)")

    async def execute(self, params: Params) -> str:  # type: ignore[override]
        try:
            async with httpx.AsyncClient(follow_redirects=True, max_redirects=5) as client:
                resp = await client.get(params.url, timeout=15.0, headers={"User-Agent": "monoclaw/1.0"})
                resp.raise_for_status()
                ct = resp.headers.get("content-type", "")
                if "html" in ct:
                    text = markdownify(resp.text, strip=["script", "style"])
                else:
                    text = resp.text
            return _truncate(text, params.max_chars)
        except Exception as exc:
            return f"fetch error: {exc}"


class ScheduleTool(Tool["ScheduleTool.Params"]):
    """Manage scheduled tasks. Actions: list/add/remove. For add, provide schedule_type, message, and timing fields."""

    def __init__(self, cfg: Config, cron: CronService) -> None:
        super().__init__(cfg)
        self._cron = cron

    class Params(BaseModel):
        action: Literal["add", "list", "remove"]
        name: str = ""
        message: str = Field(default="", description="Text to inject as user message when triggered")
        schedule_type: Literal["every", "cron", "at"] | None = None
        every_seconds: int | None = None
        cron_expr: str | None = None
        at_iso: str | None = Field(default=None, description="ISO 8601 datetime for one-shot")
        tz: str = Field(default="UTC", description="Timezone for cron_expr (default UTC)")
        job_id: str | None = Field(default=None, description="Required for 'remove'")

    @staticmethod
    def _iso_to_ms(iso: str | None) -> int | None:
        if not iso:
            return None
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    async def execute(self, params: Params) -> str:  # type: ignore[override]
        if params.action == "list":
            jobs = self._cron.list_jobs()
            if not jobs:
                return "no scheduled jobs"
            lines = [f"{j.id[:8]} | {j.name or '(unnamed)'} | {j.schedule.type} | next: {j.next_run}" for j in jobs]
            return "\n".join(lines)

        if params.action == "remove":
            if not params.job_id:
                return "job_id required for remove"
            try:
                self._cron.remove_job(params.job_id)
            except ValueError as exc:
                return f"error: {exc}"
            return f"removed job {params.job_id}"

        if params.action == "add":
            if not params.schedule_type:
                return "error: schedule_type required for add"
            stype = params.schedule_type
            schedule = CronSchedule(
                type=stype,
                every=params.every_seconds * 1000 if params.every_seconds is not None else None,
                cron_expr=params.cron_expr,
                at=self._iso_to_ms(params.at_iso),
                tz=params.tz,
            )
            try:
                job = self._cron.add_job(
                    schedule=schedule,
                    message=params.message,
                    name=params.name,
                )
            except ValueError as exc:
                return f"error: {exc}"
            return f"created job {job.id[:8]}"

        return f"unknown action: {params.action!r}"


class SetOutputChannelTool(Tool["SetOutputChannelTool.Params"]):
    """Set the output channel for this response. Call before producing output."""

    def __init__(self, cfg: Config, channel_manager: WebSocketChannelManager) -> None:
        super().__init__(cfg)
        self._channel_manager = channel_manager

    class Params(BaseModel):
        channel: str = Field(description="Channel name to route the response to")

    async def execute(self, params: Params) -> str:  # type: ignore[override]
        try:
            self._channel_manager.set_output(params.channel)
            return f"output set to {params.channel}"
        except (KeyError, ValueError) as exc:
            return f"error: {exc}"


class ListOutputChannelsTool(Tool["ListOutputChannelsTool.Params"]):
    """List available output channels."""

    def __init__(self, cfg: Config, channel_manager: WebSocketChannelManager) -> None:
        super().__init__(cfg)
        self._channel_manager = channel_manager

    class Params(BaseModel):
        pass

    async def execute(self, params: Params) -> str:  # type: ignore[override]
        names = self._channel_manager.active_channels
        if not names:
            return "no output channels available"
        return "\n".join(names)
