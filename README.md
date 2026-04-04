# monoclaw

<img src="logo.png" alt="monoclaw logo" width="100%"/>

Personal AI assistant with one continuous session, WebSocket-native multi-channel interaction, minimal codebase for clarity and security.

**codebase:** `1398 lines across 10 files`

---

- agent turn loop, tool base class + registry, file/shell/web tools, `CronService` — rewritten in Python based on [NanoClaw](https://github.com/qwibitai/nanoclaw)
- post-turn memory extraction, context window tracking and compaction — inspired by [free-code](https://github.com/paoloanzn/free-code)
- massive open-source codebase, unreviewed community contributions, security vulnerabilities nobody can trace — graciously avoided thanks to [OpenClaw](https://github.com/openclaw/openclaw)


## specific features

- **Single continuous session** — one history shared across all channels and time. The agent is coherent and persistent like a human, not stateless.
- **WebSocket-only protocol** — monoclaw speaks one protocol. Bridges (Signal, Telegram, web UI, etc.) are separate applications that connect over WebSocket and declare their name on handshake.
- **Agent-selectable output channel** — the agent can inspect active channels and redirect its reply mid-turn using tools. Default: reply to the inbound channel.
- **Automatic fire-and-forget compaction** — context is compacted automatically after the response is delivered, not before. The user never waits. History is archived before each compaction.
- **Container-as-deployment isolation** — security comes from container isolation, not application-level sandboxing. The agent process itself runs in Docker; tools operate under `data/workspace/` directly. Single WebSocket entrypoint — extension and security is shifted to proxies managed aside.

---

## Bridge protocol

A bridge connects to monoclaw via WebSocket on port `8765`.

**Handshake** (first message after connect):
```json
{"name": "signal"}
```

**Inbound message** (bridge → monoclaw):
```json
{"text": "Hello!"}
```

**Outbound chunk** (monoclaw → bridge, during generation):
```json
{"chunk": "Hello"}
```

**End of message**:
```json
{"end": true}
```

**Error** (monoclaw → bridge, e.g. bad handshake):
```json
{"error": "channel 'signal' is already connected"}
```

---

## Run

monoclaw runs as a single Docker container. Bridges run separately and connect to it.

**1. Configure** (optional — all fields have defaults; env vars also work via `LLM__BASE_URL` etc.)

```yaml
llm:
  base_url: http://your-llama-cpp-host:8080/v1
  max_tokens: 4096
```

**2. Build and run**

```bash
docker build -t monoclaw .
docker run -d \
  -p 8765:8765 \
  -v $(pwd)/config.yaml:/app/config.yaml:ro \
  -v $(pwd)/data:/app/data \
  monoclaw
```

`data/` is the persistent volume — it holds conversation history, memory, cron jobs, archives, and the agent workspace.

---

## Adding a tool

Subclass `Tool` and implement `Params` (Pydantic model) and `execute`:

```python
class MyTool(Tool["MyTool.Params"]):
    """Docstring becomes the tool's description for LLM."""
    class Params(BaseModel):
        input: str

    async def execute(self, params: Params) -> str:
        return f"result: {params.input}"
```

Register it in `ToolRegistry.from_config`. The schema is generated automatically and exposed to the LLM.

## Swapping the LLM

`LLMClient` wraps any OpenAI-compatible API. Point `llm.base_url` at any server (Ollama, vLLM, OpenAI, etc.).
