# migration_history.ps1
Write-Host "Migration history:" -ForegroundColor Cyan
docker-compose run --rm backend alembic history
