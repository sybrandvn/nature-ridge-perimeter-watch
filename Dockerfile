# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.9.16 AS uv

FROM python:3.12-slim-bookworm AS runtime

ARG APP_UID=1000
ARG APP_GID=1000

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "${APP_GID}" perimeter \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home perimeter

COPY --from=uv /uv /usr/local/bin/uv

WORKDIR /app

# Install the locked runtime environment before copying source so ordinary
# code/config changes retain the dependency layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=perimeter:perimeter config ./config
COPY --chown=perimeter:perimeter scripts ./scripts
COPY --chown=perimeter:perimeter src ./src
COPY --chown=perimeter:perimeter README.md ./README.md

RUN mkdir -p data/history data/live data/logs data/reference_bg data/reports \
    && chown -R perimeter:perimeter data

USER perimeter

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "scripts.container_healthcheck", "--readiness"]
