FROM python:3.12-slim AS runtime

ARG TORCH_BACKEND=cpu

COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN useradd --create-home --uid 10001 snake && \
    mkdir -p /app/runs && \
    chown snake:snake /app /app/runs

COPY --chown=snake:snake pyproject.toml uv.lock README.md ./

USER snake

RUN --mount=type=cache,target=/home/snake/.cache/uv,uid=10001,gid=10001 \
    uv sync --frozen --no-dev --extra train --extra ${TORCH_BACKEND} --no-install-project

COPY --chown=snake:snake src ./src
COPY --chown=snake:snake configs ./configs

RUN --mount=type=cache,target=/home/snake/.cache/uv,uid=10001,gid=10001 \
    uv sync --frozen --no-dev --extra train --extra ${TORCH_BACKEND}

VOLUME ["/app/runs"]

ENTRYPOINT ["snake-train"]
CMD ["--config", "configs/ppo.toml"]
