# debug_player_endpoint.ps1
# Script para diagnosticar el problema con el endpoint

Write-Host "🔍 DIAGNÓSTICO DEL ENDPOINT /players/{username}" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# 1. Ver logs del backend con más detalle
Write-Host "`n📋 Últimos logs del backend:" -ForegroundColor Yellow
docker logs chessplayeranalyzer-backend-1 --tail=50 2>&1 | Select-String -Pattern "ERROR|Exception|Traceback|500" -Context 2,2

# 2. Verificar que las tablas existen
Write-Host "`n📊 Verificando tablas en la BD:" -ForegroundColor Yellow
$tables = docker exec chessplayeranalyzer-postgres-1 psql -U chess -d chessdb -c "\dt" 2>$null
Write-Host $tables

# 3. Probar el endpoint directamente
Write-Host "`n🧪 Probando endpoint GET /players/test:" -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/players/test" -Method GET -ErrorAction Stop
    Write-Host "✅ Status: $($response.StatusCode)" -ForegroundColor Green
    Write-Host $response.Content
} catch {
    Write-Host "❌ Error: $($_.Exception.Message)" -ForegroundColor Red
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $errorBody = $reader.ReadToEnd()
        Write-Host "Detalles del error:" -ForegroundColor Yellow
        Write-Host $errorBody
    }
}

# 4. Verificar el endpoint raíz
Write-Host "`n🧪 Probando endpoint /:" -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/" -Method GET
    Write-Host "✅ API funcionando" -ForegroundColor Green
} catch {
    Write-Host "❌ API no responde" -ForegroundColor Red
}

# 5. Buscar el error específico en el contenedor
Write-Host "`n🔍 Buscando errores recientes en el backend:" -ForegroundColor Yellow
docker exec chessplayeranalyzer-backend-1 bash -c "grep -n 'ERROR\|Traceback' /app/*.log 2>/dev/null || echo 'No log files found'"

# 6. Verificar imports en main.py
Write-Host "`n📄 Verificando imports en main.py:" -ForegroundColor Yellow
docker exec chessplayeranalyzer-backend-1 head -20 /app/app/main.py | Select-String "import\|from"

Write-Host "`n💡 POSIBLES SOLUCIONES:" -ForegroundColor Cyan
Write-Host "1. Si falta la tabla 'player': ejecutar migraciones de Alembic" -ForegroundColor White
Write-Host "2. Si hay error de import: verificar que los modelos estén bien importados" -ForegroundColor White
Write-Host "3. Si es error de conexión BD: verificar DATABASE_URL" -ForegroundColor White