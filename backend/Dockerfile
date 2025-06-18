FROM python:3.12-slim

# Copiamos sólo lo necesario
WORKDIR /app
COPY backend/app /app/app
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Opcional pero útil: evita problemas de rutas en RUN/ENTRYPOINT
ENV PYTHONPATH=/app
