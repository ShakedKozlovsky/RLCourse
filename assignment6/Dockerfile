# Dockerfile for marl-lab — zero-setup playability for graders.
#
# Build:
#   docker build -t marl-lab assignment6/
#
# Quick smoke (CLI version):
#   docker run --rm marl-lab marl version
#   docker run --rm marl-lab marl audit
#
# Run the full audit (lint + 241 tests + LOC + graphify):
#   docker run --rm marl-lab uv run python scripts/audit.py
#
# Train + play a tiny game:
#   docker run --rm marl-lab marl train --episodes 50 --checkpoint /tmp/ckpt.pt
#   docker run --rm marl-lab marl play-game --output /tmp/report.json
#
# Multi-stage build:
#   - stage 1 ("deps"): installs dependencies — caches well, slow first build
#   - stage 2 ("runtime"): copies src + tests + assets + scripts — fast rebuild

# ---------- stage 1: dependencies ----------
FROM python:3.12-slim AS deps

# uv: fast Python package manager (used throughout this project)
COPY --from=ghcr.io/astral-sh/uv:0.5.4 /uv /usr/local/bin/uv

WORKDIR /app

# Copy only the lockfiles first so dependency layer caches well
COPY pyproject.toml uv.lock README.md ./

# Sync (incl. dev extras for pytest/ruff to support the audit subcommand)
RUN uv sync --extra dev --frozen --no-install-project

# ---------- stage 2: runtime ----------
FROM python:3.12-slim AS runtime

# uv runtime
COPY --from=ghcr.io/astral-sh/uv:0.5.4 /uv /usr/local/bin/uv

WORKDIR /app

# Copy the pre-warmed venv from the deps stage
COPY --from=deps /app/.venv /app/.venv

# Copy the project files
COPY pyproject.toml uv.lock README.md ./
COPY src/ src/
COPY tests/ tests/
COPY configs/ configs/
COPY scripts/ scripts/
COPY notebooks/ notebooks/
COPY docs/ docs/
COPY assets/ assets/

# Install the project itself (uses the cached venv)
RUN uv sync --extra dev --frozen

# Make ``uv run`` and console scripts available without explicit ``uv run`` prefix
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default entrypoint: ``docker run marl-lab marl <subcommand> ...``
# Override with ``--entrypoint`` if you want a shell or python directly.
ENTRYPOINT []
CMD ["marl", "version"]
