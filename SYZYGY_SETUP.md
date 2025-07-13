# Configuración de Tablebases Syzygy

## ¿Por qué `tb_match_rate` y `dtz_deviation` aparecen como `null`?

Los campos `tb_match_rate` y `dtz_deviation` en la sección `endgame` aparecen como `null` porque las **tablebases Syzygy** no están instaladas o configuradas en el sistema.

## ¿Qué son las Tablebases Syzygy?

Las tablebases Syzygy contienen todas las posiciones posibles con 7 piezas o menos, junto con:
- **DTM (Distance To Mate)**: Número de movimientos hasta el mate
- **DTZ (Distance To Zeroing)**: Número de movimientos hasta una posición con captura o movimiento de peón
- **WDL (Win/Draw/Loss)**: Resultado teórico de la posición

Estas tablebases permiten evaluar con precisión absoluta las posiciones de final.

## Métricas que requieren Tablebases

- **`tb_match_rate`**: Porcentaje de movimientos que coinciden con la línea óptima de la tablebase
- **`dtz_deviation`**: Desviación promedio DTZ (cuántos movimientos extra se necesitan para convertir)

## Instalación de Tablebases Syzygy

### 1. Descargar las Tablebases

Las tablebases Syzygy están disponibles en: https://tablebase.sesse.net/syzygy/

**Opciones de descarga:**
- **3-4-5 piezas**: ~1.5 GB (básico)
- **3-4-5-6 piezas**: ~150 GB (recomendado)
- **3-4-5-6-7 piezas**: ~18 TB (completo, solo para servidores)

### 2. Configurar el Directorio

```bash
# Crear directorio para las tablebases
mkdir -p /data/syzygy

# Descomprimir los archivos .rtbw y .rtbz en este directorio
# Los archivos deben estar directamente en /data/syzygy/, no en subdirectorios
```

### 3. Configurar la Variable de Entorno

```bash
# En el archivo .env o en el entorno del sistema
export SYZYGY_PATH="/data/syzygy"
```

### 4. Verificar la Instalación

```bash
# Verificar que los archivos están presentes
ls -la /data/syzygy/*.rtbw | head -5
ls -la /data/syzygy/*.rtbz | head -5

# Verificar que el directorio es accesible
python -c "from pathlib import Path; print(Path('/data/syzygy').exists())"
```

## Estructura de Archivos Esperada

```
/data/syzygy/
├── KPK.rtbw
├── KPK.rtbz
├── KRK.rtbw
├── KRK.rtbz
├── KQK.rtbw
├── KQK.rtbz
├── KNNK.rtbw
├── KNNK.rtbz
├── KBBK.rtbw
├── KBBK.rtbz
├── KNBK.rtbw
├── KNBK.rtbz
├── KRNK.rtbw
├── KRNK.rtbz
├── KQNK.rtbw
├── KQNK.rtbz
├── KQBK.rtbw
├── KQBK.rtbz
├── KQRK.rtbw
├── KQRK.rtbz
└── ... (más archivos para 4-5-6-7 piezas)
```

## Configuración en Docker

Si usas Docker, necesitas montar el directorio de tablebases:

```yaml
# docker-compose.yml
version: '3.8'
services:
  chess-analyzer:
    # ... otras configuraciones
    volumes:
      - /path/to/syzygy:/data/syzygy:ro
    environment:
      - SYZYGY_PATH=/data/syzygy
```

## Verificación

Después de la instalación, ejecuta un análisis de jugador y verifica que:

1. Los logs muestren: `"DEBUG ENDGAME: Tablebase path exists, analyzing with Syzygy"`
2. Los campos `tb_match_rate` y `dtz_deviation` ya no sean `null`
3. Los valores sean números reales (porcentajes para `tb_match_rate`, números para `dtz_deviation`)

## Notas Importantes

- **Espacio en disco**: Las tablebases completas requieren mucho espacio
- **Tiempo de análisis**: Con tablebases, el análisis será más lento pero más preciso
- **Memoria**: Las tablebases se cargan en memoria, asegúrate de tener suficiente RAM
- **Permisos**: El proceso debe tener permisos de lectura en el directorio de tablebases

## Alternativas

Si no puedes instalar las tablebases completas:
- Usa solo las tablebases de 3-4-5 piezas (~1.5 GB)
- Las métricas `conversion_efficiency` funcionarán sin tablebases
- Las métricas de `tb_match_rate` y `dtz_deviation` aparecerán como `null` pero no afectarán otras funcionalidades 