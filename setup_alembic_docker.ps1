# setup_alembic_docker.ps1
# Script para configurar Alembic dentro del contenedor Docker en Windows

Write-Host "🔧 CONFIGURANDO ALEMBIC EN DOCKER" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

# 1. Verificar que el contenedor esté corriendo
Write-Host "`n🐳 Verificando contenedores..." -ForegroundColor Yellow
$runningContainers = docker ps --format "table {{.Names}}" | Select-String "backend"

if (-not $runningContainers) {
    Write-Host "❌ El contenedor backend no está corriendo" -ForegroundColor Red
    Write-Host "   Ejecuta primero: docker-compose up -d" -ForegroundColor White
    exit 1
}

# 2. Instalar Alembic en el contenedor
Write-Host "`n📦 Instalando Alembic en el contenedor..." -ForegroundColor Yellow
docker exec chessplayeranalyzer-backend-1 pip install alembic

# 3. Inicializar Alembic
Write-Host "`n🎯 Inicializando Alembic..." -ForegroundColor Yellow
docker exec chessplayeranalyzer-backend-1 bash -c "cd /app && alembic init alembic"

# 4. Crear alembic.ini
Write-Host "`n⚙️ Creando alembic.ini..." -ForegroundColor Yellow

$alembicIni = @'
[alembic]
script_location = alembic
prepend_sys_path = .
version_path_separator = os
sqlalchemy.url = postgresql+psycopg://chess:chess@postgres:5432/chessdb

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
'@

# Guardar temporalmente y copiar al contenedor
$tempFile = New-TemporaryFile
Set-Content -Path $tempFile.FullName -Value $alembicIni -Encoding UTF8
docker cp $tempFile.FullName chessplayeranalyzer-backend-1:/app/alembic.ini
Remove-Item $tempFile.FullName

# 5. Crear env.py
Write-Host "`n📝 Creando env.py..." -ForegroundColor Yellow

$envPy = @'
from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
import os
import sys

# Agregar app al path
sys.path.insert(0, '/app')

# Importar modelos
from app.models import SQLModel
from app import models

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = SQLModel.metadata

def get_url():
    return os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://chess:chess@postgres:5432/chessdb"
    )

def run_migrations_offline() -> None:
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section)
    configuration['sqlalchemy.url'] = get_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
'@

# Guardar temporalmente y copiar al contenedor
$tempFile = New-TemporaryFile
Set-Content -Path $tempFile.FullName -Value $envPy -Encoding UTF8
docker cp $tempFile.FullName chessplayeranalyzer-backend-1:/app/alembic/env.py
Remove-Item $tempFile.FullName

# 6. Actualizar requirements.txt
Write-Host "`n📄 Actualizando requirements.txt..." -ForegroundColor Yellow
docker exec chessplayeranalyzer-backend-1 bash -c "echo 'alembic>=1.13.0' >> requirements.txt"

Write-Host "`n✅ ALEMBIC CONFIGURADO EXITOSAMENTE" -ForegroundColor Green

# 7. Crear scripts auxiliares
Write-Host "`n📝 Creando scripts auxiliares..." -ForegroundColor Yellow

# Script para crear migraciones
$migrationScript = @'
# create_migration.ps1
param(
    [Parameter(Mandatory=$true)]
    [string]$Message
)

Write-Host "Creating migration: $Message" -ForegroundColor Cyan
docker-compose run --rm backend alembic revision --autogenerate -m "$Message"
'@
Set-Content -Path "create_migration.ps1" -Value $migrationScript

# Script para aplicar migraciones
$upgradeScript = @'
# upgrade_db.ps1
Write-Host "Applying database migrations..." -ForegroundColor Cyan
docker-compose run --rm backend alembic upgrade head
'@
Set-Content -Path "upgrade_db.ps1" -Value $upgradeScript

# Script para ver historial
$historyScript = @'
# migration_history.ps1
Write-Host "Migration history:" -ForegroundColor Cyan
docker-compose run --rm backend alembic history
'@
Set-Content -Path "migration_history.ps1" -Value $historyScript

Write-Host "`n✨ CONFIGURACIÓN COMPLETADA" -ForegroundColor Green
Write-Host "`n📋 COMANDOS DISPONIBLES:" -ForegroundColor Yellow
Write-Host "  - Crear migración: .\create_migration.ps1 -Message 'Add detailed analysis tables'" -ForegroundColor White
Write-Host "  - Aplicar migraciones: .\upgrade_db.ps1" -ForegroundColor White
Write-Host "  - Ver historial: .\migration_history.ps1" -ForegroundColor White
Write-Host "`n💡 También puedes usar docker-compose directamente:" -ForegroundColor Yellow
Write-Host "  docker-compose run --rm backend alembic revision --autogenerate -m 'mensaje'" -ForegroundColor Gray
Write-Host "  docker-compose run --rm backend alembic upgrade head" -ForegroundColor Gray