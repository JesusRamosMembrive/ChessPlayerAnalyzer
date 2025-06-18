FROM python:3.12-slim
WORKDIR /app

# 1) Sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends stockfish postgresql-client && \
    ln -s /usr/games/stockfish /usr/local/bin/stockfish && \
    rm -rf /var/lib/apt/lists/*

# 2) Dependencias Python
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) Código
COPY backend/  /app

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
