import asyncio
import http
import json
import time
from collections.abc import Callable, Coroutine

from typing import Any

import websockets
from pydantic import BaseModel, ValidationError

from config import CRON_CHANNEL, WS_HOST, WS_MAX_FRAME_BYTES, WS_PORT, logger

HANDSHAKE_OPTIONS = {"classifier": True, "quick_session": False}  # the defaults (see the protocol below)


class InboundImage(BaseModel):
    mime: str
    data: str  # base64 payload, without the "data:<mime>;base64," prefix
    name: str = ""


class InboundMessage(BaseModel):
    channel: str
    text: str
    timestamp: int
    images: list[InboundImage] = []
    # Everything said in reply, in order (fast classifier + agent), and whether the turn is
    # over — the next message's fast-classifier context.
    replies: list[str] = []
    answered: bool = False


class WebSocketChannelManager:
    """
    WebSocket server where each connected client is a named channel.

    Protocol:
      - Client connects and sends a handshake with its name: {"name": "signal/bob"}, optionally with
        "classifier": false (no fast classifier: every message goes to the agent; default true) and
        "quick_session": true (a fresh conversation: no history loaded or saved, memory extracted once
        when the connection closes, and a turn still running then is cancelled; default false)
      - Client sends messages: {"text": "..."} with optional
        "images": [{"mime": "image/jpeg", "data": "<base64>", "name": "photo.jpg"}]
        (an entry may also be an MP4 video: "mime": "video/mp4") and "interrupt": true (cancel the
        channel's running and queued turns first)
      - Server sends replies as two frames: {"chunk": "..."} then {"end": true}; a message can get
        several replies (progress lines end with {"end": true, "interim": true}, then the answer), and
        {"done": true} follows the last one

    Plain HTTP `GET /status` on the same port answers {"busy": bool}: a message is being
    classified or answered, or a cron turn runs. modeswitch waits for false before swapping the LLM.
    """

    def __init__(self) -> None:
        self._connections: dict[str, Any] = {}
        self._options: dict[str, dict[str, bool]] = {}  # per connected channel: the handshake's options
        self._tasks: dict[str, set[asyncio.Task]] = {}  # per channel: its messages still being handled
        self._on_message: Callable[[InboundMessage], Coroutine[Any, Any, None]]
        self._agent_busy: Callable[[], bool]
        self._on_connect: Callable[[str, dict[str, bool]], None]
        self._on_disconnect: Callable[[str, dict[str, bool]], None]
        self._in_flight = 0

    # Lifecycle

    async def start(self, on_message: Callable[[InboundMessage], Coroutine[Any, Any, None]],
                    agent_busy: Callable[[], bool], on_connect: Callable[[str, dict[str, bool]], None],
                    on_disconnect: Callable[[str, dict[str, bool]], None]) -> None:
        self._on_message = on_message
        self._agent_busy = agent_busy
        self._on_connect, self._on_disconnect = on_connect, on_disconnect
        logger.info(f"websocket server listening on {WS_HOST}:{WS_PORT}")
        async with websockets.serve(self._handle, WS_HOST, WS_PORT, max_size=WS_MAX_FRAME_BYTES,
                                    process_request=self._status):
            await asyncio.Future()

    def _status(self, connection: Any, request: Any) -> Any:
        if request.path != "/status":
            return None                     # proceed with the WebSocket handshake
        busy = self._in_flight > 0 or self._agent_busy()
        return connection.respond(http.HTTPStatus.OK, json.dumps({"busy": busy}) + "\n")

    async def _dispatch(self, msg: InboundMessage) -> None:
        self._in_flight += 1
        try:
            await self._on_message(msg)
        finally:
            self._in_flight -= 1

    @property
    def active_channels(self) -> list[str]:
        return list(self._connections.keys())

    def classifier_enabled(self, channel_name: str) -> bool:
        return self._options.get(channel_name, {}).get("classifier", True)

    # Sending — all methods raise if the channel isn't connected or the send fails.

    async def send_chunk(self, channel_name: str, text: str, images: list[dict] | None = None) -> None:
        """Send a single chunk to a channel (no end frame). Use for streaming.

        `images` are delivered as attachments by the bridge. Omitted from the frame when empty, so
        text-only traffic stays byte-identical to before.
        """
        ws = self._connections.get(channel_name)
        if ws is None:
            raise RuntimeError(f"channel {channel_name!r} is not connected")
        frame: dict = {"chunk": text}
        if images:
            frame["images"] = images
        await ws.send(json.dumps(frame))

    async def end_msg(self, channel_name: str, interim: bool = False) -> None:
        """Signal end-of-message to a channel; interim: a progress line, not the answer."""
        ws = self._connections.get(channel_name)
        if ws is None:
            raise RuntimeError(f"channel {channel_name!r} is not connected")
        await ws.send(json.dumps({"end": True, "interim": True} if interim else {"end": True}))

    async def send_full_msg(self, channel_name: str, text: str, images: list[dict] | None = None,
                            interim: bool = False) -> None:
        """Atomic chunk + end convenience — single complete message."""
        await self.send_chunk(channel_name, text, images)
        await self.end_msg(channel_name, interim)

    async def turn_done(self, channel_name: str) -> None:
        """Signal that a message has had all its replies (a client that ignores unknown frames, like
        signal-bridge, is unaffected)."""
        ws = self._connections.get(channel_name)
        if ws is None:
            raise RuntimeError(f"channel {channel_name!r} is not connected")
        await ws.send(json.dumps({"done": True}))

    # WebSocket handler

    async def _handle(self, ws: Any) -> None:
        try:
            raw = await ws.recv()
            handshake = json.loads(raw)
            name: str = handshake.get("name", "").strip()
            if not name:
                logger.warning("handshake rejected: missing name")
                await ws.send(json.dumps({"error": "handshake JSON must include a non-empty 'name'"}))
                await ws.close()
                return
            if name in self._connections or name == CRON_CHANNEL:
                logger.warning(err := f"handshake rejected: channel <{name!r}> is not available")
                await ws.send(json.dumps({"error": err}))
                await ws.close()
                return
            options = {k: handshake.get(k, default) for k, default in HANDSHAKE_OPTIONS.items()}
            if bad := [k for k, v in options.items() if not isinstance(v, bool)]:
                logger.warning(err := f"handshake rejected: {', '.join(bad)} must be true or false")
                await ws.send(json.dumps({"error": err}))
                await ws.close()
                return
        except Exception as exc:
            logger.warning(f"handshake failed: {exc}")
            await ws.close()
            return

        self._connections[name], self._options[name] = ws, options
        logger.info(f"🔌 channel connected: {name!r}"
                    + "".join(f", {k}={v}" for k, v in options.items() if v != HANDSHAKE_OPTIONS[k]))
        self._on_connect(name, options)

        try:
            async for raw in ws:
                try:
                    data = json.loads(raw)
                except Exception as exc:
                    logger.warning(f"invalid message from {name!r}: {exc}")
                    await ws.send(json.dumps({"error": f"invalid JSON: {exc}"}))
                    continue
                text: str = data.get("text", "")
                try:
                    msg = InboundMessage(
                        channel=name,
                        text=text.strip(),
                        timestamp=int(time.time() * 1000),
                        images=data.get("images") or [],
                    )
                except ValidationError as exc:
                    logger.warning(f"invalid message from {name!r}: {exc}")
                    await ws.send(json.dumps({"error": f"invalid message: {exc}"}))
                    continue
                # never log msg.images — base64 payloads would flood the log
                img_note = f" +{len(msg.images)} image(s)" if msg.images else ""
                logger.info(f"📨 message from {name!r}: {text[:60]!r}{img_note}")
                if data.get("interrupt") is True:
                    self._cancel(name, "interrupted by a newer message")
                tasks = self._tasks.setdefault(name, set())
                tasks.add(task := asyncio.create_task(self._dispatch(msg)))
                task.add_done_callback(tasks.discard)
        finally:
            self._connections.pop(name, None)
            logger.info(f"🔌 channel disconnected: {name!r}")
            options = self._options.pop(name)
            if options["quick_session"]:
                self._cancel(name, "the quick session closed")  # nobody is left to answer
            self._on_disconnect(name, options)

    def _cancel(self, channel_name: str, why: str) -> None:
        """Cancel the channel's messages still being handled: the running turn and any queued behind it."""
        if tasks := [t for t in self._tasks.get(channel_name, ()) if not t.done()]:
            logger.info(f"✋ cancelling {len(tasks)} turn(s) of {channel_name!r}: {why}")
            for task in tasks:
                task.cancel()
