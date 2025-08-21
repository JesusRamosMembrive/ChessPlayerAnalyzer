[ACTUALIZACIÓN — 2025-08-21]
Estado de sugerencias tras los últimos cambios:
- Ya implementado (no repetir aquí): ACPL robusto con cap en quality.py; trazabilidad/metadatos y robustez en timing.py; guardas en longitudinal.py para evitar inestabilidades con una sola partida.
- Sugerencias que siguen vigentes (a continuación) permanecen como mejoras futuras opcionales.

# Sugerencias adicionales (no críticas, positivas para el proyecto)

Estas propuestas no corrigen “errores” pero mejoran la calidad, interpretabilidad y estabilidad de los cálculos en `app/analysis/*`.

## 1) Métrica de “pérdida robusta” adicional

- Añadir, además de ACPL, una métrica de “pérdida robusta” basada en mediana o media recortada (trimmed mean al 10-20%) sobre `delta_eval` capado. Esto complementa ACPL y es menos sensible a outliers.

## 2) Métrica WDL-ACPL (escala acotada por victoria/empate/derrota)

- Mapear evaluaciones a probabilidad WDL (Win/Draw/Loss) y definir una pérdida en [0, 1] relativa a la mejor jugada según WDL. Evita la explosión de mates y centra la medida en impacto práctico en resultado.

## 3) Separar reportes por fase de forma estandarizada

- Devolver sistemáticamente un bloque `phase_quality` con:
  - `opening_acpl`, `middlegame_acpl`, `endgame_acpl` (robustos)
  - `opening_blunder_rate`, `middlegame_blunder_rate`, `endgame_blunder_rate`
- Útil para diagnósticos y para no sobreinterpretar outliers de fin de partida.

## 4) Normalización y rangos esperados en quality_score

- Antes de combinar ACPL con `match_rate` y `weighted_match_rate`, aplicar normalizaciones a [0,1] con saturaciones razonables (p. ej., ACPL capado a 100 cp para el score). Documentar el rango objetivo del score para facilitar thresholds consistentes.

## 5) Métrica “second_choice_rate” a nivel de jugada y fase

- Si ya existe la señal a nivel de partidas, exponer el cálculo a nivel de movimientos y fases (porcentaje de veces que el jugador elige la PV[2] cuando PV[1] y PV[0] están muy próximas). Señal útil para detectar “jugar alrededor” de la primera línea.

## 6) Calidad y tiempo: features combinadas

- Crear una función en `app/analysis` que combine selectivamente quality+timing (p. ej., `clutch_accuracy` + ACPL robusto en ventanas) para estudiar “mejora súbita tras pausa”. Esto queda en análisis y no toca la adquisición.

## 7) Chequeos de sanidad previos/posteriores

- Añadir en `aggregate_quality_features()` y `aggregate_time_features()` logs de sanidad:
  - % de filas válidas para cálculo de ACPL/`match_rate`/corr.
  - Rango cuantiles (p10-p90) de `delta_eval`/`move_time`.
  - Conteo de jugadas con PV de mate si el DF trae bandera; si no, derivar heurística simple (picos > cap).

## 8) Configurabilidad por parámetros

- Hacer configurables (por kwargs con defaults) los caps, trims, umbrales de blunder, etc., de forma que los experimentos de calibración no requieran tocar código.

## 9) Reproducibilidad y auditoría

- Incluir en los resultados de `aggregate_quality_features()` un pequeño anexo de “meta” (p. ej., `{acpl_cap_used: 1500, used_delta_eval: true, rows_used: n, rows_total: N}`) para auditar decisiones de cálculo en cada partida.

## 10) Tipado y compatibilidad

- Evitar `.values` en NDArray cuando sea ambiguo; preferir `.to_numpy(dtype=float)`.
- Añadir anotaciones de tipos explícitas (PEP 484/PEP 561) para reducir falsos positivos del analizador estático. No cambia lógica, mejora DX y estabilidad.

Implementar estas mejoras en `app/analysis` aumentará robustez de las métricas, reducirá falsos positivos/negativos y hará más explicables los resultados para revisores humanos.
