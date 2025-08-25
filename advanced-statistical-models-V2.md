# Advanced Statistical Models – V2

## 1. Calibración Bayesiana basada en datos reales
Afinar los priors con estadísticas empíricas por rango Elo y experiencia, usando técnicas bayesianas jerárquicas.

## 2. Calibración de probabilidades del ensemble
Aplicar isotonic regression o Platt scaling tras combinar Bayesian + ML para obtener probabilidades bien calibradas.

## 3. Autoencoder LSTM para anomalías
Completar el stub `lstm_autoencoder_scores` para capturar secuencias de movimientos y tiempos, generando un `reconstruction_error`.

## 4. Modelos GARCH para volatilidad
Extender `performance_model.py` con un modelo GARCH que modele la varianza condicional de ACPL/ROI.

## 5. SPC multivariante en tiempo real
Añadir Hotelling’s T² y MEWMA para monitorizar métricas correlacionadas y soportar análisis “streaming”.
