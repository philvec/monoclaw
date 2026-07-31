import asyncio
import json
import mimetypes
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

import httpx
from pydantic import BaseModel, Field, ValidationError

from channels import WebSocketChannelManager
from config import Config, IMAGE_MIME_EXT, IMAGES_DIR, logger, sniff_image_mime, to_decodable_image
from llm import LLMClient
from mcp_client import note_mcp_result
from memory_store import MemoryStore

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
        self._mcp_client: Any = None
        self._mcp_schemas: list[dict] = []
        self._mcp_names: set[str] = set()
        self.current_channel: str | None = None
        self._last_file: str | None = None

    @classmethod
    def from_config(
        cls,
        cfg: Config,
        cron: CronService,
        channel_manager: WebSocketChannelManager,
        memory_store: MemoryStore,
        llm: LLMClient,
    ) -> "ToolRegistry":
        registry = cls()
        tools: list[Tool[Any]] = [
            ReadFileTool(cfg),
            ViewImageTool(cfg),
            FetchImageTool(cfg),
            WriteFileTool(cfg),
            EditFileTool(cfg),
            GlobTool(cfg),
            GrepTool(cfg),
            ShellTool(cfg),
            ScheduleTool(cfg, cron),
            SendMessageTool(cfg, channel_manager),
            ListChannelsTool(cfg, channel_manager),
            DeferTurnTool(cfg, cron),
            MemorySearchTool(cfg, memory_store, llm),
            MemoryReadTool(cfg, memory_store),
            MasterMemoryTool(cfg, memory_store),
        ]
        for tool in tools:
            registry.register(tool)
        return registry

    def attach_mcp(self, client: Any) -> None:
        self._mcp_client = client
        self._mcp_schemas = client.tool_schemas
        self._mcp_names = {s["function"]["name"] for s in self._mcp_schemas}

    def register(self, tool: Tool[Any]) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool[Any]:
        return self._tools[name]

    @property
    def definitions(self) -> list[dict]:
        return [t.to_schema() for t in self._tools.values()] + self._mcp_schemas

    async def execute(self, name: str, arguments: dict) -> str:
        if name == "send_message" and self.current_channel is not None:
            if arguments.get("channel") == self.current_channel:
                return (
                    f"error: cannot send_message to the current input channel ({self.current_channel!r}) — "
                    "reply through the standard Answer instead"
                )
        if name in self._tools:
            tool = self._tools[name]
            # Track last accessed file so we can hint on missing-path errors
            if "path" in arguments and isinstance(arguments["path"], str):
                self._last_file = arguments["path"]
            try:
                result = await tool.execute(tool.Params.model_validate(arguments))
            except ValidationError as exc:
                logger.exception(f"tool {name!r} raised")
                missing = [str(e["loc"][0]) for e in exc.errors() if e["type"] == "missing"]
                if missing:
                    hint = f" (last file used: {self._last_file!r})" if self._last_file and "path" in missing else ""
                    return (
                        f"error: {name} missing required parameter(s): {', '.join(missing)}{hint}. "
                        f"Retry with ALL required fields: "
                        + ", ".join(tool.Params.model_json_schema().get("required", []))
                    )
                return f"error: {exc}"
            except Exception as exc:
                logger.exception(f"tool {name!r} raised")
                return f"error: {exc}"
            if isinstance(result, str):
                return result
            try:
                return json.dumps(result)
            except (TypeError, ValueError):
                return str(result)
        if self._mcp_client is not None and name in self._mcp_names:
            ok, text = await self._mcp_client.call_checked(name, arguments)
            note_mcp_result(ok, text, self.current_channel)  # may restart the process
            return text
        return f"unknown tool: {name!r}"


def _safe_path(rel: str) -> Path:
    """Resolve rel relative to _WORKSPACE; raise ValueError if it escapes."""
    base = _WORKSPACE.resolve()
    target = (base / rel).resolve()
    try:
        target.relative_to(base)
    except ValueError:
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


class ViewImageTool(Tool["ViewImageTool.Params"]):
    """Show an image file that is already in the workspace — e.g. one a script just generated.
    The picture is added to your context so you can check what it shows, AND the filename it returns
    goes into Answer.attachments to send that file to the user. THIS is the tool for showing a local
    file; fetch_image is only for downloading from an http(s) URL."""

    # mtmd decodes via stb_image; IMAGE_MIME_EXT is the single source of truth for what we handle.
    _MIMES = set(IMAGE_MIME_EXT)

    class Params(BaseModel):
        path: str = Field(description="Relative path to an image file (jpg, png, webp, gif, bmp)")

    async def execute(self, params: Params) -> str:  # type: ignore[override]
        target = _safe_path(params.path)
        if not target.is_file():
            # A "[IMAGE <file> <mime>]" marker names a stored image, not a workspace file. The model
            # does sometimes reference one anyway; resolve it instead of hard-failing.
            stored = IMAGES_DIR / Path(params.path).name  # basename only — must not escape IMAGES_DIR
            if not stored.is_file():
                return f"image not found: {params.path}"
            # Already stored, and already attached to the message it came from: re-showing it would
            # duplicate the picture in context for no gain.
            stored_mime = mimetypes.guess_type(stored.name)[0] or sniff_image_mime(stored.read_bytes()[:16])
            return f"[IMAGE {stored.name} {stored_mime or 'image/jpeg'}]"
        mime = mimetypes.guess_type(target.name)[0] or ""
        if mime not in self._MIMES:
            mime = sniff_image_mime(target.read_bytes()[:16])  # e.g. .webp is absent from mimetypes
        if mime not in self._MIMES:
            return f"not a supported image: {params.path} (detected {mime or 'unknown'})"
        # Copy into IMAGES_DIR so the marker resolves for the rest of the conversation even if the
        # workspace file is later moved or overwritten.
        try:
            data, mime = to_decodable_image(target.read_bytes(), mime)
        except Exception as exc:
            return f"error: could not decode {params.path} ({type(exc).__name__})"
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        stamp = int(datetime.now(timezone.utc).timestamp() * 1000)
        fname = f"{stamp}-{target.stem}{IMAGE_MIME_EXT.get(mime) or '.img'}"
        (IMAGES_DIR / fname).write_bytes(data)
        logger.info(f"🖼️ view_image: {params.path} → {fname}")
        # Marker must lead — agent._expand_markers only treats leading lines as image markers.
        return (
            f"[IMAGE {fname} {mime}]\n(showing {params.path})\n"
            f'To send this file to the user, set attachments to ["{fname}"] — exactly that string.'
        )


class FetchImageTool(Tool["FetchImageTool.Params"]):
    """Download an image from an http(s) URL on the web and look at it. The picture is added to your
    context, so you can check what it actually shows before deciding to send it; the returned filename
    goes in Answer.attachments to deliver it to the user. Web URLs ONLY — for a file that is already
    in the workspace (e.g. one you just generated), use view_image instead."""

    _MAX_BYTES = 8_000_000

    class Params(BaseModel):
        url: str = Field(description="Direct URL of an image file (jpg, png, webp, gif, bmp)")

    async def execute(self, params: Params) -> str:  # type: ignore[override]
        if not params.url.lower().startswith(("http://", "https://")):
            # Observed: the model reaches here with file:///workspace/x.png or a localhost URL when it
            # wants to show a file it just made. Name the right tool instead of just refusing.
            return (
                f"error: not an http(s) url: {params.url!r}. fetch_image only downloads from the web — "
                "to show a file that is already in the workspace, call view_image with its relative path."
            )
        async with httpx.AsyncClient(follow_redirects=True, max_redirects=5) as client:
            resp = await client.get(params.url, timeout=20.0, headers={"User-Agent": "monoclaw/1.0"})
            resp.raise_for_status()
            mime = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
            if mime not in ViewImageTool._MIMES:
                return f"error: {params.url} is {mime or 'unknown type'}, not a supported image"
            if len(resp.content) > self._MAX_BYTES:
                return f"error: image is {len(resp.content) / 1e6:.1f} MB (limit {self._MAX_BYTES / 1e6:.0f} MB)"
            data = resp.content
        try:
            data, mime = to_decodable_image(data, mime)  # e.g. webp → png; never store undecodable
        except Exception as exc:
            return f"error: could not decode {params.url} ({type(exc).__name__}) — try another URL"
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        stamp = int(datetime.now(timezone.utc).timestamp() * 1000)
        fname = f"{stamp}-fetched{IMAGE_MIME_EXT.get(mime) or '.img'}"
        (IMAGES_DIR / fname).write_bytes(data)
        logger.info(f"🖼️ fetch_image: {params.url} → {fname} ({len(data) / 1e3:.0f} kB)")
        # Marker must lead — agent._expand_markers only treats leading lines as image markers.
        # The exact attachments value is spelled out because the model otherwise invents a
        # descriptive filename ("wawel-castle-krakow.jpg") instead of copying this opaque one.
        return (
            f"[IMAGE {fname} {mime}]\n(fetched from {params.url})\n"
            f'To send this picture, set attachments to ["{fname}"] — exactly that string, do not rename it.'
        )


class WriteFileTool(Tool["WriteFileTool.Params"]):
    """Write content to a file inside the workspace, creating it if needed. This only SAVES the file —
    it does not execute anything. Running is a different tool: use shell for that."""

    class Params(BaseModel):
        path: str = Field(description="Relative path to file")
        content: str

    async def execute(self, params: Params) -> str:  # type: ignore[override]
        target = _safe_path(params.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(params.content)
        out = f"wrote {len(params.content)} chars to {params.path}"
        if target.suffix in (".py", ".sh"):
            # Saying WHY (write_file cannot execute) rather than forbidding the workaround: it kept
            # writing a subprocess-based "runner", i.e. reaching for execution through the only tool
            # it thought it had.
            runner = "python" if target.suffix == ".py" else "bash"
            out += (
                f"\nSaved, not executed — write_file cannot run anything. "
                f"To run it, use the shell tool: `{runner} {params.path}`"
            )
        return out


class EditFileTool(Tool["EditFileTool.Params"]):
    """Replace an exact string in a file. REQUIRED: path, old_string, new_string."""

    class Params(BaseModel):
        path: str = Field(description="Relative path to file (REQUIRED)")
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
    """Run a shell command; the working directory is the workspace. This is the only tool that
    EXECUTES anything: write_file saves a file, shell runs it (e.g. `python draw.py` after writing
    draw.py). Also use it to check what is available (`python -c "import PIL"`)."""

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
            lines = [f"{j.id} | {j.name or '(unnamed)'} | {j.schedule.type} | next: {j.next_run}" for j in jobs]
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


class DeferTurnTool(Tool["DeferTurnTool.Params"]):
    """Schedule a future turn so YOU regain initiative without waiting for a user message. \
Use this after sending a message when you want to follow up later, check on something, or \
continue a thread after a delay (e.g. you reminded the wife to take meds — defer 30 min and \
report back to the user). The `note` becomes the user message that fires the turn. \
For recurring chores (daily reports, periodic checks), use `schedule` instead — this is for \
one-shot self-wakeups that complete a workflow."""

    def __init__(self, cfg: Config, cron: CronService) -> None:
        super().__init__(cfg)
        self._cron = cron

    class Params(BaseModel):
        note: str = Field(
            description="Reminder text injected as the user message when the turn fires "
            "(e.g. 'follow up: did wife confirm she took meds?')"
        )
        delay_seconds: int | None = Field(
            default=None, description="Wake up after this many seconds from now (alternative to at_iso)"
        )
        at_iso: str | None = Field(
            default=None, description="ISO 8601 datetime to wake up at, e.g. '2026-04-14T18:30:00+02:00'"
        )

    async def execute(self, params: Params) -> str:  # type: ignore[override]
        if params.delay_seconds is None and not params.at_iso:
            return "error: provide delay_seconds or at_iso"
        if params.at_iso:
            dt = datetime.fromisoformat(params.at_iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            at_ms = int(dt.timestamp() * 1000)
        else:
            at_ms = int(datetime.now(timezone.utc).timestamp() * 1000) + (params.delay_seconds or 0) * 1000
        try:
            job = self._cron.add_job(
                schedule=CronSchedule(type="at", at=at_ms),
                message=params.note,
                name="defer_turn",
            )
        except ValueError as exc:
            return f"error: {exc}"
        return f"deferred turn scheduled (job {job.id[:8]})"


class SendMessageTool(Tool["SendMessageTool.Params"]):
    """Send a complete message to a named channel — FAN-OUT ONLY. Use this to notify someone \
on a DIFFERENT channel from the inbound one (e.g. cc the wife while replying to a friend). \
You do NOT need this tool to reply to the sender of the current turn — that reply is produced \
simply by writing it as your assistant content, and the runtime auto-delivers it to the INPUT \
CHANNEL. This tool is for reaching *other* recipients in the same turn."""

    def __init__(self, cfg: Config, channel_manager: WebSocketChannelManager) -> None:
        super().__init__(cfg)
        self._channel_manager = channel_manager

    class Params(BaseModel):
        channel: str = Field(
            description="Target channel name (must be currently connected). Should differ from INPUT CHANNEL."
        )
        text: str = Field(description="Exact message text the recipient will see")

    async def execute(self, params: Params) -> str:  # type: ignore[override]
        try:
            await self._channel_manager.send_full_msg(params.channel, params.text)
        except Exception as exc:
            return f"error: {exc}"
        return f"sent to {params.channel}"


class ListChannelsTool(Tool["ListChannelsTool.Params"]):
    """List currently connected channels you can target with `send_message` for fan-out. \
Use before fanning out if you're not sure who's reachable."""

    def __init__(self, cfg: Config, channel_manager: WebSocketChannelManager) -> None:
        super().__init__(cfg)
        self._channel_manager = channel_manager

    class Params(BaseModel):
        pass

    async def execute(self, params: Params) -> str:  # type: ignore[override]
        names = self._channel_manager.active_channels
        if not names:
            return "no channels connected"
        return "\n".join(names)


class MemorySearchTool(Tool["MemorySearchTool.Params"]):
    """Search long-term memories. Returns ranked results with snippets. Hybrid keyword + semantic search."""

    def __init__(self, cfg: Config, store: MemoryStore, llm: LLMClient) -> None:
        super().__init__(cfg)
        self._store = store
        self._llm = llm

    class Params(BaseModel):
        query: str = Field(description="Search keywords or phrase")
        limit: int = Field(default=10, description="Max results to return")
        type: str = Field(default="", description="Filter by type: user/project/reference/feedback/skill (empty = all)")

    async def execute(self, params: Params) -> str:  # type: ignore[override]
        query_embedding = await self._llm.embed(params.query)
        mem_type = params.type if params.type else None
        results = self._store.search(
            params.query, query_embedding=query_embedding, limit=params.limit, mem_type=mem_type
        )
        if not results:
            return "no matching memories"
        lines = []
        for r in results:
            lines.append(f"**{r.slug}** [{r.type}] (score: {r.score:.2f})")
            lines.append(f"  {r.snippet}")
            lines.append("")
        return "\n".join(lines).strip()


class MemoryReadTool(Tool["MemoryReadTool.Params"]):
    """Read the full content of a specific memory by its slug."""

    def __init__(self, cfg: Config, store: MemoryStore) -> None:
        super().__init__(cfg)
        self._store = store

    class Params(BaseModel):
        slug: str = Field(description="Memory slug identifier")

    async def execute(self, params: Params) -> str:  # type: ignore[override]
        entry = self._store.get(params.slug)
        if entry is None:
            return f"memory not found: {params.slug}"
        return (
            f"**{entry.slug}** [{entry.type}]\n"
            f"Created: {entry.created.isoformat()} | Updated: {entry.updated.isoformat()}\n\n"
            f"{entry.content}"
        )


class MasterMemoryTool(Tool["MasterMemoryTool.Params"]):
    """Read or update master memory — core identity and instructions always in the system prompt. \
Use for: user name, language, timezone, key behavioral rules, critical context. \
Only update when the user explicitly asks. Changes affect every future turn."""

    def __init__(self, cfg: Config, store: MemoryStore) -> None:
        super().__init__(cfg)
        self._store = store

    class Params(BaseModel):
        action: Literal["read", "write"] = Field(description="'read' to view, 'write' to replace")
        content: str = Field(default="", description="New content (only for write)")

    async def execute(self, params: Params) -> str:  # type: ignore[override]
        if params.action == "read":
            text = self._store.read_master_memory()
            return text if text else "(master memory is empty)"
        self._store.write_master_memory(params.content)
        return f"master memory updated ({len(params.content)} chars)"
