import asyncio
import json
import time
from collections.abc import Callable, Coroutine

from typing import Any

import websockets
from pydantic import BaseModel, ValidationError

from config import CRON_CHANNEL, WS_HOST, WS_MAX_FRAME_BYTES, WS_PORT, logger


class InboundImage(BaseModel):
    mime: str
    data: str  # base64 payload, without the "data:<mime>;base64," prefix
    name: str = ""


class InboundMessage(BaseModel):
    channel: str
    text: str
    timestamp: int
    images: list[InboundImage] = []


class WebSocketChannelManager:
    """
    WebSocket server where each connected client is a named channel.

    Protocol:
      - Client connects and sends a handshake with its name: {"name": "signal/bob"}
      - Client sends messages: {"text": "..."} with optional
        "images": [{"mime": "image/jpeg", "data": "<base64>", "name": "photo.jpg"}]
      - Server sends replies as two frames: {"chunk": "..."} then {"end": true}
    """

    def __init__(self) -> None:
        self._connections: dict[str, Any] = {}
        self._on_message: Callable[[InboundMessage], Coroutine[Any, Any, None]]

    # Lifecycle

    async def start(self, on_message: Callable[[InboundMessage], Coroutine[Any, Any, None]]) -> None:
        self._on_message = on_message
        logger.info(f"websocket server listening on {WS_HOST}:{WS_PORT}")
        async with websockets.serve(self._handle, WS_HOST, WS_PORT, max_size=WS_MAX_FRAME_BYTES):
            await asyncio.Future()

    @property
    def active_channels(self) -> list[str]:
        return list(self._connections.keys())

    # Sending — all methods raise if the channel isn't connected or the send fails.

    async def send_chunk(self, channel_name: str, text: str) -> None:
        """Send a single chunk to a channel (no end frame). Use for streaming."""
        ws = self._connections.get(channel_name)
        if ws is None:
            raise RuntimeError(f"channel {channel_name!r} is not connected")
        await ws.send(json.dumps({"chunk": text}))

    async def end_msg(self, channel_name: str) -> None:
        """Signal end-of-message to a channel."""
        ws = self._connections.get(channel_name)
        if ws is None:
            raise RuntimeError(f"channel {channel_name!r} is not connected")
        await ws.send(json.dumps({"end": True}))

    async def send_full_msg(self, channel_name: str, text: str) -> None:
        """Atomic chunk + end convenience — single complete message."""
        await self.send_chunk(channel_name, text)
        await self.end_msg(channel_name)

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
        except Exception as exc:
            logger.warning(f"handshake failed: {exc}")
            await ws.close()
            return

        self._connections[name] = ws
        logger.info(f"🔌 channel connected: {name!r}")

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
                asyncio.create_task(self._on_message(msg))
        finally:
            self._connections.pop(name, None)
            logger.info(f"🔌 channel disconnected: {name!r}")
