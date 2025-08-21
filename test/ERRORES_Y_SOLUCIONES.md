# Informe de errores y soluciones (centrado en app/analysis)

Este documento resume los problemas detectados en los cálculos/metodología y propone soluciones concretas priorizando cambios en `app/analysis/*`. Solo se sugiere tocar `test/run_local_analysis.py` si hay fallos intrínsecos en sí mismo, pero el foco está en `app/analysis`.

Fuentes:
- Logs: `test/result2Text2.txt`
- Cálculo de calidad: `app/analysis/quality.py`
- Cálculo de tiempos: `app/analysis/timing.py`

## 1) ACPL mal definido (swing de evaluación, no “pérdida vs mejor jugada”)

- Evidencia:
  - En `quality.acpl()` se define ACPL como media del valor absoluto de `eval_cp_after - eval_cp_before` (ajustado por color). Esto mide “cambio de eval” y NO la distancia respecto a la mejor jugada del motor.
  - Consecuencia: penaliza jugadas fuertes que cambian mucho la evaluación a favor del jugador y puede no penalizar jugadas malas que apenas cambian la evaluación.
- Impacto observado: El DataFrame usado por calidad incluye un `delta_eval` extremo (99491 en el movimiento 17) y el ACPL calculado deriva en ~5600.44, arrastrando IPR y quality_score a valores absurdos.
- Solución propuesta (en `app/analysis/quality.py`):
  - Cambiar la base de ACPL a “pérdida vs mejor jugada” cuando el DataFrame aporte la columna `delta_eval` que representa precisamente esa pérdida (distancia a PV[0] del motor).
  - Implementar en `quality.acpl()` la siguiente lógica:
    - Si existe `delta_eval`, usar `mean(abs(delta_eval))` como ACPL.
    - Si no existe `delta_eval`, mantener el fallback actual a `mean(abs(eval_cp_after - eval_cp_before))`.
  - Esto permite que la definición canónica (pérdida contra mejor jugada) viva en `app/analysis`, independientemente de cómo se haya creado el DF.
- Cambios concretos sugeridos:
  - `app/analysis/quality.py::acpl()`:
    - Detectar la columna `delta_eval` y priorizarla.
    - Mantener ajuste por color solo si se usa el fallback de eval_before/after.
  - `app/analysis/quality.py::aggregate_quality_features()`:
    - Sin cambios de firma, pero el valor retornado de `acpl()` representará la métrica correcta si existe `delta_eval`.

## 2) Desbordamiento por mates: mate_score y PV de mate inflan ACPL/IPR

- Evidencia:
  - Los logs muestran `delta_eval=99491` en el movimiento 17 (mate cerca), y el ACPL ≈ 5600.44. Esto genera IPR ≈ -666.89 y quality_score ≈ -2191.68.
- Causa:
  - Evaluaciones de mate se traducen a centipawns muy grandes (p. ej. 100000) y contaminan agregados.
- Solución propuesta (en `app/analysis/quality.py`):
  - Introducir mecanismos de robustez al calcular ACPL y derivados:
    1) Capar los valores de pérdida para ACPL, p. ej. `delta_eval_capped = clip(delta_eval, -1500, 1500)` antes de promediar.
    2) Excluir de ACPL las jugadas con PV que contienen “mate” (si el DF trae un flag o inferencia). Si no llega ese flag desde el DF, exponer un parámetro opcional para cap universal (solución 1).
    3) Alternativamente, usar un agregador robusto (mediana o media recortada 10-20%) para ACPL.
  - Recomendación concreta mínima y segura: aplicar cap simétrico (p. ej. 1500 cp) a los valores usados para ACPL dentro de `quality.acpl()` cuando exista `delta_eval`. Esto evita que una sola jugada terminal destroce el promedio.
- Cambios concretos sugeridos:
  - `quality.acpl()`:
    - Si usa `delta_eval`, aplicar cap configurable, p. ej. `cap_cp=1500` (parámetro opcional con valor por defecto).
    - Documentar que el cap solo se aplica a inputs marcadamente extremos, no a las diferencias normales.
  - `aggregate_quality_features()`:
    - Sin cambios, heredará el ACPL robusto.

## 3) Cohesión con best_rank: cp_loss/ACPL debe basarse en la mejor PV

- Evidencia:
  - El rank (`best_rank`) y `is_engine_best` se derivan de MultiPV. Sin embargo, el ACPL actual en `quality` ignora la relación con la mejor PV si no usa `delta_eval`.
- Solución propuesta (en `app/analysis/quality.py`):
  - Asegurar que ACPL utilice `delta_eval` cuando está disponible (ver punto 1). Así queda alineado con `best_rank`/`is_engine_best` que provienen de MultiPV.
  - Si `delta_eval` no está disponible, añadir un aviso en logs para informar que se está usando el fallback (swing de eval), menos interpretativo.

## 4) ACPL por fase y blunders: estabilidad y granularidad

- Evidencia:
  - El DF ya tiene columna `phase` (opening/middlegame/endgame). Jugadas terminales/endgame con grandes swings sesgan el ACPL global.
- Solución propuesta (en `app/analysis/quality.py`):
  - Añadir en `aggregate_quality_features()` métricas por fase (p. ej. `opening_acpl`, `middlegame_acpl`, `endgame_acpl`) reusando una función auxiliar interna que compute ACPL por fase usando `delta_eval` capado cuando esté.
  - Alternativamente, devolver junto con `acpl` una versión “acpl_nm” (no-terminal moves) que excluya PV de mate si la información existe en DF (o un cap alto).
  - Ya existe `compute_phase_quality()` en `quality.py` para listas de DataFrames; aprovechar su criterio para el caso por-partida si el DF ya está etiquetado.
- Cambios concretos sugeridos:
  - Extender `aggregate_quality_features()` para incluir, si hay `phase`, un sub-bloque de métricas por fase (sin romper compatibilidad).
  - O bien, dejar `aggregate_quality_features()` como está y exponer una función `phase_acpl(game_df)` en `quality.py`, de uso por capas superiores.

## 5) quality_score dependiente de ACPL: hacerla robusta

- Evidencia:
  - `quality_score` combina ACPL, `match_rate` y `weighted_match_rate`. Si ACPL se desborda, el score se hace negativo o absurdo.
- Solución propuesta (en `app/analysis/quality.py`):
  - Una vez ACPL sea robusto (cap/mediana), `quality_score` deja de colapsar. Aun así, se recomienda:
    - Normalizar ACPL antes de mezclarlo, p. ej. `scaled_acpl = 1 - min(acpl, 100)/100` para que esté en [0,1] y no explote.
    - Documentar el rango esperado y el efecto del cap en la métrica.

## 6) Reproducibilidad depth/time en análisis de calidad

- Evidencia:
  - Los logs muestran `go depth 12 movetime 5000` repetido. Mezclar un tiempo fijo con un depth objetivo puede dar variaciones si el tiempo no alcanza, o añadir ruido en distintas máquinas.
- Solución propuesta (en `app/analysis/quality.py` – a nivel de interpretación):
  - No se cambia engine aquí, pero se puede:
    - Registrar en los logs de `quality` el depth alcanzado (si llega al DF) y advertir si es inferior al objetivo esperado. Esto ayuda a interpretar ACPL/`best_rank`.
  - El cambio de engine y límites pertenece a la etapa de enriquecimiento; como el alcance es `app/analysis`, proponemos solo mejorar la trazabilidad en cálculos (logging).

## 7) Fallos del motor: filas con error no deben contribuir a ACPL

- Evidencia:
  - En el run analizado no se registran fallos, pero la metodología general debería prevenir sesgos.
- Solución propuesta (en `app/analysis/quality.py`):
  - Si existen columnas/flags en el DF que indiquen “eval no disponible/NaN”, excluir esas filas de `acpl()` y de agregados (`match_rate`, etc.). Evitar imputar 0 por defecto.
  - Añadir logs de conteo de filas excluidas.

## 8) Tipos y robustez (diagnósticos estáticos)

- Evidencia:
  - pyright/pylance advierte sobre tipos en `ACPLModel` y en `.values` sobre `NDArray`.
- Solución propuesta (en `app/analysis/quality.py`):
  - Ajustar los tipos en llamadas a `HuberRegressor` y `.predict`, y evitar `.values` si el tipo no es Series; usar `.to_numpy()` con `dtype=float` donde aplique.
  - Esto no cambia métricas, solo estabilidad del módulo.

---

## Resumen de cambios propuestos (todos en `app/analysis/quality.py`):

1) `acpl(game_df, player_color='white', cap_cp: int | None = 1500)`
   - Si `delta_eval` existe: usar `delta_eval` (cap si `cap_cp` no es None).
   - Si no: fallback a `abs(eval_cp_after - eval_cp_before)` ajustado por color.
   - Excluir filas inválidas/NaN y loguear conteos.

2) `aggregate_quality_features(game_df, elo: int | None, player_color='white')`
   - Aprovecha el ACPL robusto y conserva API.
   - Opcional: añadir ACPL por fase si `phase` existe.
   - Calcular `quality_score` con ACPL robusto y/o normalizado a rango.

3) Opcional: nueva util `phase_acpl(game_df, cap_cp=1500)`
   - Devuelve dict con ACPL por fase usando `delta_eval` con cap y conteos.

4) Ajustes menores de tipos/logging en `ACPLModel` y funciones auxiliares.

Con estos cambios, el sistema medirá la pérdida real frente a la mejor jugada (cuando el DF lo soporte), será robusto ante mates/terminales y evitará que un único outlier destruya las métricas derivadas.
