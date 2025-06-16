# fix_alembic_ini.ps1
# Script para crear un alembic.ini válido sin BOM

Write-Host "🔧 Arreglando alembic.ini..." -ForegroundColor Cyan

# Contenido del alembic.ini
$alembicContent = @'
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

# Guardar sin BOM usando UTF8 sin BOM
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText("backend\alembic.ini", $alembicContent, $utf8NoBom)

Write-Host "✅ alembic.ini guardado localmente sin BOM" -ForegroundColor Green

# Si Alembic no está inicializado, inicializarlo primero
Write-Host "`n🔍 Verificando si Alembic está inicializado..." -ForegroundColor Yellow
$alembicExists = docker exec chessplayeranalyzer-backend-1 test -d /app/alembic
if ($LASTEXITCODE -ne 0) {
    Write-Host "📦 Inicializando Alembic..." -ForegroundColor Yellow
    docker exec chessplayeranalyzer-backend-1 alembic init alembic
}

# Ahora crear el env.py correcto
Write-Host "`n📝 Creando env.py..." -ForegroundColor Yellow

$envContent = @'
from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
import os
import sys

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import models
from app.models import SQLModel
from app import models  # This imports all model classes

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

# Guardar env.py sin BOM
[System.IO.File]::WriteAllText("backend\alembic\env.py", $envContent, $utf8NoBom)

Write-Host "✅ env.py guardado sin BOM" -ForegroundColor Green

Write-Host "`n🚀 Archivos arreglados. Ahora puedes ejecutar:" -ForegroundColor Cyan
Write-Host "   docker-compose run --rm backend alembic revision --autogenerate -m 'Add detailed analysis tables'" -ForegroundColor White