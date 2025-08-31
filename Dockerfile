# syntax=docker/dockerfile:1.6
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl stockfish && rm -rf /var/lib/apt/lists/*
WORKDIR /app

FROM base AS deps-base
COPY requirements-base.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -U pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-base.txt && \
    pip uninstall -y chess || true && pip uninstall -y python-chess || true && \
    pip install --no-cache-dir python-chess==1.999
ENV PYTHONPATH=/app

FROM deps-base AS app-base
COPY . /app
ENV PATH="/root/.local/bin:${PATH}"

FROM python:${PYTHON_VERSION}-slim AS torch

WORKDIR /app
COPY requirements-ml.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-ml.txt
COPY --from=deps-base /usr/local /usr/local
COPY . /app
ENV PYTHONPATH=/app
