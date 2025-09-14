"""
Funciones helper para análisis extraídas de engine.py.
Reduce la complejidad del motor principal.
"""
from __future__ import annotations

import logging
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from app.utils.data_processing import aggregate_basic_metrics, safe_divide

logger = logging.getLogger(__name__)


def calculate_suspicion_score(features: Dict, rating: Optional[int], experience: int) -> float:
    """
    Combina modelos bayesianos y supervisados para un score final.
    Extraído de engine._calculate_suspicion_score para reutilización.
    """
    try:
        # Intentar importar modelos de ML
        from app.analysis.bayesian import BayesianSuspicionModel
        from app.analysis.ml_classifier import MLSuspicionClassifier

        # --- Modelo bayesiano -------------------------------------------------
        bayes_model = BayesianSuspicionModel()
        evidence = {
            'acpl': features.get('acpl', 0),
            'match_rate': features.get('match_rate', 0),
            'time_complexity_corr': features.get('time_complexity_corr', 0),
            'lag_spike_count': features.get('lag_spike_count', 0),
            'opening_entropy': features.get('H_opening', 0),
            'second_choice_rate': features.get('second_choice_pct', 0),
        }
        bayes_prob = bayes_model.update(rating, experience, evidence)

        # --- Clasificador supervisado ---------------------------------------
        ml_prob = bayes_prob
        try:
            ml_clf = MLSuspicionClassifier()
            raw_ml_prob = ml_clf.predict_proba(features)
            ml_prob = ml_clf.calibrate_prob(bayes_prob, raw_ml_prob)
        except Exception as exc:
            logger.warning("ML classifier unavailable: %s", exc)

        # Soft voting: media de ambas probabilidades
        final_prob = (bayes_prob + ml_prob) / 2
        return float(final_prob)

    except ImportError as e:
        logger.warning(f"ML models not available: {e}")
        # Fallback simple basado en métricas básicas
        return _simple_suspicion_fallback(features)


def _simple_suspicion_fallback(features: Dict) -> float:
    """
    Fallback simple para calcular sospecha sin modelos ML.
    """
    # Thresholds empíricos
    acpl = features.get('acpl', 100)  # Average centipawn loss
    match_rate = features.get('match_rate', 0.5)  # Match rate con engine

    # Score básico: menor acpl y mayor match_rate = más sospechoso
    acpl_score = max(0, (50 - acpl) / 50)  # Normalizar: <50 acpl es sospechoso
    match_score = max(0, (match_rate - 0.7) / 0.3)  # >70% match rate es sospechoso

    return min(1.0, (acpl_score + match_score) / 2)


def calculate_basic_risk_score(games_metrics: Dict[str, np.ndarray],
                               long_features: Dict) -> Tuple[float, Dict]:
    """
    Calcula un score de riesgo básico 0-100 usando NumPy arrays.
    Versión optimizada de engine._calculate_risk_score.
    """
    risk_factors = {}
    risk_components = []

    # 1. Análisis de precisión (ACPL)
    if 'acpl' in games_metrics:
        acpl_values = games_metrics['acpl']
        acpl_stats = aggregate_basic_metrics(acpl_values)
        avg_acpl = acpl_stats['mean']

        if not np.isnan(avg_acpl):
            # ACPL < 15 es muy sospechoso para jugadores no-GM
            acpl_risk = max(0, (25 - avg_acpl) / 25) * 100
            risk_factors['acpl_risk'] = acpl_risk
            risk_components.append(('acpl', acpl_risk, 0.3))

    # 2. Análisis de match rate
    if 'match_rate' in games_metrics:
        match_values = games_metrics['match_rate']
        match_stats = aggregate_basic_metrics(match_values)
        avg_match = match_stats['mean']

        if not np.isnan(avg_match):
            # Match rate > 85% es sospechoso
            match_risk = max(0, (avg_match - 0.75) / 0.25) * 100
            risk_factors['match_rate_risk'] = match_risk
            risk_components.append(('match_rate', match_risk, 0.25))

    # 3. Análisis temporal
    if 'time_complexity_corr' in games_metrics:
        time_corr_values = games_metrics['time_complexity_corr']
        time_stats = aggregate_basic_metrics(time_corr_values)
        avg_time_corr = time_stats['mean']

        if not np.isnan(avg_time_corr):
            # Correlación muy baja entre tiempo y complejidad es sospechosa
            time_risk = max(0, (0.3 - avg_time_corr) / 0.3) * 100
            risk_factors['timing_risk'] = time_risk
            risk_components.append(('timing', time_risk, 0.2))

    # 4. Consistencia anormal
    consistency = long_features.get('consistency_score', 0.5)
    if isinstance(consistency, (int, float)) and not np.isnan(consistency):
        # Consistencia muy alta es sospechosa
        consistency_risk = max(0, (consistency - 0.8) / 0.2) * 100
        risk_factors['consistency_risk'] = consistency_risk
        risk_components.append(('consistency', consistency_risk, 0.15))

    # 5. Patrones de apertura
    opening_entropy = long_features.get('opening_patterns', {}).get('entropy', 2.0)
    if isinstance(opening_entropy, (int, float)) and not np.isnan(opening_entropy):
        # Entropía muy baja = repetitivo, sospechoso
        entropy_risk = max(0, (1.5 - opening_entropy) / 1.5) * 100
        risk_factors['opening_risk'] = entropy_risk
        risk_components.append(('opening', entropy_risk, 0.1))

    # Calcular score final ponderado
    if risk_components:
        total_weight = sum(weight for _, _, weight in risk_components)
        weighted_score = sum(score * weight for _, score, weight in risk_components)
        final_risk = weighted_score / total_weight if total_weight > 0 else 0
    else:
        final_risk = 0

    # Aplicar bonificaciones/penalizaciones
    game_count = len(games_metrics.get('acpl', []))
    if game_count < 10:
        # Pocos juegos = menos confiable
        final_risk *= 0.7
        risk_factors['low_sample_penalty'] = True
    elif game_count > 50:
        # Muchos juegos = más confiable
        final_risk *= 1.1
        risk_factors['high_sample_bonus'] = True

    return min(100.0, max(0.0, final_risk)), risk_factors


def normalize_game_features(games_data: List[Dict]) -> Dict[str, np.ndarray]:
    """
    Convierte lista de diccionarios de juegos a arrays NumPy normalizados.
    """
    if not games_data:
        return {}

    # Extraer todas las claves posibles
    all_keys = set()
    for game in games_data:
        all_keys.update(game.keys())

    result = {}

    for key in all_keys:
        values = []
        for game in games_data:
            value = game.get(key, np.nan)
            # Convertir strings numéricos
            if isinstance(value, str):
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    value = np.nan
            values.append(value)

        result[key] = np.array(values, dtype=np.float32)

    return result


def detect_statistical_anomalies(values: np.ndarray, method: str = 'iqr') -> Dict[str, Any]:
    """
    Detecta anomalías estadísticas en arrays de valores.
    """
    if len(values) == 0:
        return {'outliers': np.array([], dtype=bool), 'outlier_count': 0, 'outlier_rate': 0.0}

    valid_values = values[~np.isnan(values)]
    if len(valid_values) < 4:
        return {'outliers': np.zeros_like(values, dtype=bool), 'outlier_count': 0, 'outlier_rate': 0.0}

    if method == 'iqr':
        q1 = np.percentile(valid_values, 25)
        q3 = np.percentile(valid_values, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = (values < lower_bound) | (values > upper_bound)
    elif method == 'zscore':
        mean_val = np.mean(valid_values)
        std_val = np.std(valid_values)
        if std_val > 0:
            z_scores = np.abs((values - mean_val) / std_val)
            outliers = z_scores > 3
        else:
            outliers = np.zeros_like(values, dtype=bool)
    else:
        outliers = np.zeros_like(values, dtype=bool)

    outlier_count = np.sum(outliers)
    outlier_rate = outlier_count / len(values) if len(values) > 0 else 0.0

    return {
        'outliers': outliers,
        'outlier_count': int(outlier_count),
        'outlier_rate': float(outlier_rate),
        'bounds': (float(lower_bound), float(upper_bound)) if method == 'iqr' else None
    }


def aggregate_performance_metrics(games_metrics: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Agrega métricas de performance de múltiples juegos.
    """
    result = {}

    # Métricas básicas de calidad
    quality_metrics = ['acpl', 'match_rate', 'weighted_match_rate', 'ipr', 'ipr_z_score']
    for metric in quality_metrics:
        if metric in games_metrics:
            stats = aggregate_basic_metrics(games_metrics[metric])
            result.update({
                f"{metric}_mean": stats['mean'],
                f"{metric}_std": stats['std'],
                f"{metric}_median": stats['median']
            })

    # Métricas temporales
    timing_metrics = ['mean_move_time', 'time_variance', 'time_complexity_corr', 'uniformity_score']
    for metric in timing_metrics:
        if metric in games_metrics:
            stats = aggregate_basic_metrics(games_metrics[metric])
            result.update({
                f"{metric}_mean": stats['mean'],
                f"{metric}_std": stats['std']
            })

    # Métricas de apertura
    opening_metrics = ['opening_entropy', 'novelty_depth', 'second_choice_rate']
    for metric in opening_metrics:
        if metric in games_metrics:
            stats = aggregate_basic_metrics(games_metrics[metric])
            result.update({
                f"{metric}_mean": stats['mean']
            })

    # Correlaciones importantes
    if 'acpl' in games_metrics and 'match_rate' in games_metrics:
        from app.utils.data_processing import correlation_coefficient
        result['acpl_match_correlation'] = correlation_coefficient(
            games_metrics['acpl'], games_metrics['match_rate']
        )

    return result