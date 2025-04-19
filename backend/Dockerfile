FROM python:3.12-slim

WORKDIR /code

# 1) Sistema: Stockfish
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        stockfish postgresql-client && \
    ln -s /usr/games/stockfish /usr/local/bin/stockfish && \
    rm -rf /var/lib/apt/lists/*

# 2) Dependencias Python
COPY requirements.txt /code/
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copiamos el resto del proyecto (incluye entrypoint.sh)
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]