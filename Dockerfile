FROM python:3.12-slim

# Copiamos sólo lo necesario
WORKDIR /app
COPY app /app/app
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY app/analysis/eco_table.json /app/app/analysis/
RUN apt-get update && apt-get install -y stockfish


RUN pip install -U pip setuptools wheel && \
    pip uninstall -y chess python-chess && \
    pip install --no-cache-dir python-chess==1.999

# Opcional pero útil: evita problemas de rutas en RUN/ENTRYPOINT
ENV PYTHONPATH=/app
