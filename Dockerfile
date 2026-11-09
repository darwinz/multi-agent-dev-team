FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore PYTHONUNBUFFERED=1
# Run as an unprivileged user so files in the mounted workspace aren't root-owned on the host
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd -g "$APP_GID" coder && useradd -m -u "$APP_UID" -g "$APP_GID" coder \
    && mkdir /app && chown coder:coder /app
WORKDIR /app

# Dependencies first (cached layer), from the lockfile export (`make requirements`), then the project itself.
# Installed system-wide as root: exec_python runs `python -s`, which ignores user site-packages.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=coder:coder . .
# Editable: the BST grading fixture (tests/fixtures), the API collection and web/ are read from the source tree
RUN pip install --no-cache-dir --no-deps -e .
RUN python -c "from app.demo_check import FIXTURE_PATH; from app.web import WEB_DIR; from app.api_check import DEFAULT_COLLECTION; assert FIXTURE_PATH.is_file() and WEB_DIR.is_dir() and DEFAULT_COLLECTION.is_file(), 'source-tree assets missing'"

USER coder
# Bake Chroma's default local embedding model into the image (no download at first run)
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; DefaultEmbeddingFunction()(['warmup'])"

ENV CODER_WEB_HOST=0.0.0.0
EXPOSE 8000
ENTRYPOINT ["team"]
CMD ["--demo", "bst", "--reset-memory"]
