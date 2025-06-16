# backend/Dockerfile
FROM python:3.11-slim

# Instalar Stockfish y dependencias del sistema
RUN apt-get update && apt-get install -y \
    stockfish \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements y instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY app/ ./app/

# Variables de entorno por defecto
ENV PYTHONUNBUFFERED=1
ENV STOCKFISH_PATH=/usr/games/stockfish

# Puerto por defecto
EXPOSE 8000

# Comando por defecto (se sobrescribe en docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]