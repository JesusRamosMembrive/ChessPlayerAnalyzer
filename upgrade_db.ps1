# upgrade_db.ps1
Write-Host "Applying database migrations..." -ForegroundColor Cyan
docker-compose run --rm backend alembic upgrade head
