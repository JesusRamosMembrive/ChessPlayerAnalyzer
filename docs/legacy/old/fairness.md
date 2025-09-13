# Fairness Monitoring

El sistema calcula métricas de sesgo para las predicciones de los modelos:

- **Demographic Parity**: diferencia máxima en la tasa de positivos entre grupos.
- **Equalized Odds**: diferencia máxima de TPR y FPR entre grupos.

Umbrales recomendados:

| Métrica | Umbral |
|---------|--------|
| Demographic Parity | ≤ 0.10 |
| Equalized Odds (TPR/FPR) | ≤ 0.10 |

Durante el entrenamiento, el pipeline valida automáticamente que las
métricas anteriores no superen estos límites y lanza un error en caso
contrario.
