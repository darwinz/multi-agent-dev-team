FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /uvx /bin/
# Run as an unprivileged user so files in the mounted workspace aren't root-owned on the host
ARG UID=1000
ARG GID=1000
RUN groupadd -g "$GID" coder && useradd -m -u "$UID" -g "$GID" coder \
    && mkdir /app && chown coder:coder /app
USER coder
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependencies first (cached layer), then the project itself
COPY --chown=coder:coder pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project
COPY --chown=coder:coder . .
RUN uv sync --frozen --no-dev

# Bake Chroma's default local embedding model into the image (no download at first run)
RUN uv run --no-sync python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; DefaultEmbeddingFunction()(['warmup'])"

ENV CODER_WEB_HOST=0.0.0.0 PATH=/app/.venv/bin:$PATH
EXPOSE 8000
CMD ["coder-web"]
