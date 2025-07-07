# Copias de seguridad y recuperación de la base de datos

Este proyecto incluye una utilidad CLI para generar copias de seguridad de la base de datos **PostgreSQL** y restaurarlas de forma sencilla.

## Requisitos

* El cliente oficial de PostgreSQL (`pg_dump`, `psql`, `gzip`, `gunzip`) debe estar disponible en el contenedor o máquina donde se ejecute el comando.
* Credenciales de acceso configuradas mediante `DATABASE_URL` o variables estándar `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, `PGDATABASE`.

## Crear una copia de seguridad

Dentro del contenedor **backend** (o cualquier entorno donde esté instalado el código):

```bash
python -m app.db_backup backup                    # Genera ./db_backups/<db>_<timestamp>.sql.gz
```

Opciones:

* `--out DIR`  – Directorio de destino (por defecto `./db_backups`)
* `--no-compress` – Desactiva la compresión gzip (genera `.sql` plano)

El archivo resultante contiene TODO el esquema y los datos de la base, listo para ser restaurado en otra instancia de PostgreSQL.

## Restaurar una copia de seguridad

```bash
python -m app.db_backup restore ./db_backups/chessdb_20250703_153012.sql.gz
```

El script realizará los siguientes pasos:

1. Verificar la existencia de la base de datos destino; si no existe, la crea.
2. Importar todo el contenido del archivo (`gunzip` on-the-fly si está comprimido).

> **Nota:** El proceso sobrescribirá cualquier dato existente en las tablas. Úsalo con precaución.

## Ejecución vía `docker-compose`

Para automatizar backups programados puedes lanzar un contenedor efímero:

```bash
docker compose run --rm backend python -m app.db_backup backup --out /app/archives/db_backups
```

Los archivos estarán disponibles en el volumen compartido `archives/` dentro de tu directorio de proyecto en el host.

## Restaurar desde un contenedor Postgres externo

Si el dump proviene de otro servidor o necesitas restaurar sobre un contenedor **postgres** distinto:

```bash
# Copia el dump dentro del contenedor postgres y ejecútalo allí
cat chessdb_20250703_153012.sql.gz | \ 
  docker exec -i <nombre_contenedor_postgres> bash -c \
  "gunzip -c | psql -U $POSTGRES_USER -d $POSTGRES_DB"
```

Asegúrate de reemplazar `<nombre_contenedor_postgres>` por el identificador real.

## Buenas prácticas

* Almacena backups en una ubicación externa y segura.
* Automatiza la generación de copias diarias mediante cron o GitHub Actions.
* Verifica periódicamente la integridad restaurando en un entorno de staging. 