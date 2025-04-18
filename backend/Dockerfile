# backend/Dockerfile
FROM python:3.12-slim

WORKDIR /code

# 1) Dependencias del sistema (stockfish + compilers)
RUN apt-get update && \
    apt-get install -y --no-install-recommends stockfish && \
    ln -s /usr/games/stockfish /usr/local/bin/stockfish && \
    rm -rf /var/lib/apt/lists/*

# 2) Python deps
COPY requirements.txt /code/
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copiamos el resto del proyecto
COPY . /code

# 4) Comando por defecto se define en docker‑compose (api/worker)
