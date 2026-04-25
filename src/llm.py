import asyncio
from typing import Any

import httpx
import json_repair
import numpy as np
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from config import LLMConfig, logger


def _parse_args(raw: str | None) -> dict:
    """Repair and parse a (potentially malformed) JSON arguments string into a dict."""
    try:
        result = json_repair.loads(raw or "{}")
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict


class LLMResponse(BaseModel):
    content: str
    finish_reason: str  # "stop" | "tool_calls" | "length" | "error"
    tool_calls: list[ToolCall] = []
    parsed: BaseModel | None = None
    input_tokens: int = 0
    error: str = ""


class LLMClient:
    def __init__(self, cfg: LLMConfig) -> None:
        self._cfg = cfg
        self._schema_tools: list[dict] = []
        self._client = AsyncOpenAI(
            base_url=cfg.base_url,
            api_key="sk-local",  # required by AsyncOpenAI; llama.cpp ignores it
        )

    def set_schema_tools(self, tools: list[dict]) -> None:
        self._schema_tools = tools

    async def fetch_context_window(self) -> int:
        """Fetch the context window size, retrying until the model is ready."""
        while True:
            try:
                models = await self._client.models.list()
                model = next(iter(models.data), None)
                ctx = (model.model_extra or {}).get("context_length") if model else None
                return int(ctx) if ctx else 8192
            except Exception as exc:
                logger.info(f"waiting for LLM to be ready: {exc}")
                await asyncio.sleep(5)

    async def embed(self, text: str) -> np.ndarray | None:
        """Generate embedding via /v1/embeddings. Uses embeddings_url if set, else base_url."""
        url = (self._cfg.embeddings_url or self._cfg.base_url).rstrip("/")
        try:
            async with httpx.AsyncClient() as http:
                resp = await http.post(
                    f"{url}/embeddings",
                    json={"input": text, "model": "local"},
                    timeout=10.0,
                )
                if resp.status_code != 200:
                    logger.warning(f"embedding endpoint returned {resp.status_code}")
                    return None
                data = resp.json()
                return np.array(data["data"][0]["embedding"], dtype=np.float32)
        except Exception as exc:
            logger.warning(f"embedding generation failed: {exc}")
            return None

    async def chat(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[dict] | None = None,
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": "local",  # llama.cpp ignores this field
            "messages": messages,
            "max_tokens": max_tokens or self._cfg.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "extra_body": {"chat_template_kwargs": {"enable_thinking": self._cfg.enable_thinking}},
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if response_model is not None and not tools:
            # response_format=json_schema suppresses tool calls in llama.cpp — only use it
            # when there are no real tools so the model is free to call tools when needed.
            schema = response_model.model_json_schema()
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": response_model.__name__, "schema": schema},
            }
            if self._schema_tools:
                kwargs["tools"] = self._schema_tools
                kwargs["tool_choice"] = "auto"

        chunks: list[Any] = []
        try:
            stream = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream:
                chunks.append(chunk)
        except Exception as exc:
            return LLMResponse(content="", finish_reason="error", error=str(exc))

        result = self._parse_chunks(chunks)
        if response_model is not None and result.content:
            parsed: BaseModel | None = None
            try:
                parsed = response_model.model_validate_json(result.content)
            except Exception:
                try:
                    # Extract the JSON object from the content — handles preamble text,
                    # markdown code fences, and any trailing prose.
                    raw = result.content
                    start = raw.find("{")
                    end = raw.rfind("}") + 1
                    if start != -1 and end > start:
                        raw = raw[start:end]
                    repaired = json_repair.loads(raw)
                    if isinstance(repaired, dict):
                        parsed = response_model.model_validate(repaired)
                except Exception as exc:
                    logger.warning(f"structured output parse failed: {exc}")
            if parsed is None:
                logger.warning(f"parse produced None: finish={result.finish_reason!r} content={result.content[:300]!r}")
            result = result.model_copy(update={"parsed": parsed})
        return result

    def _parse_chunks(self, chunks: list[Any]) -> LLMResponse:
        content_parts: list[str] = []
        finish_reason = "stop"
        tool_call_accum: dict[int, dict] = {}  # index → partial tool call
        input_tokens = 0

        for chunk in chunks:
            if not chunk.choices:
                # usage chunk
                if hasattr(chunk, "usage") and chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens or 0
                continue

            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason

            delta = choice.delta
            if delta.content:
                content_parts.append(delta.content)

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_call_accum:
                        tool_call_accum[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc_delta.id:
                        tool_call_accum[idx]["id"] += tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_call_accum[idx]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_call_accum[idx]["arguments"] += tc_delta.function.arguments

        tool_calls = []
        for v in tool_call_accum.values():
            tool_calls.append(ToolCall(id=v["id"] or "0", name=v["name"], arguments=_parse_args(v["arguments"])))

        return LLMResponse(
            content="".join(content_parts),
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
        )
