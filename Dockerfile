FROM python:3.12.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN chmod -R a+rX /app/src
RUN uv sync --locked --no-dev

RUN useradd --system app
USER app

EXPOSE 8000
CMD ["/app/.venv/bin/python", "-m", "support_agent.mcp_server"]
