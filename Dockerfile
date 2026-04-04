# monoclaw agent container
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash curl wget git ripgrep jq ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv --no-cache-dir

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src/ ./src/

RUN mkdir -p /app/data/workspace /app/data/memory /app/data/archive

EXPOSE 8765

CMD ["uv", "run", "--no-dev", "python", "src/main.py"]