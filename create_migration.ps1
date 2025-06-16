# create_migration.ps1
param(
    [Parameter(Mandatory=$true)]
    [string]$Message
)

Write-Host "Creating migration: $Message" -ForegroundColor Cyan
docker-compose run --rm backend alembic revision --autogenerate -m "$Message"
