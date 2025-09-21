"""
Tests unitarios para módulo de análisis de calidad.
"""
import pytest
from app.analysis.quality import (
    calculate_acpl,
    calculate_match_rate,
    classify_move_quality,
    aggregate_quality_features
)


class TestCalculateACPL:
    """Tests para cálculo de ACPL (Average Centipawn Loss)."""

    def test_acpl_perfect_game(self):
        """Test ACPL con partida perfecta (sin pérdidas)."""
        moves_data = [
            {"cp_loss": 0},
            {"cp_loss": 0},
            {"cp_loss": 0}
        ]
        result = calculate_acpl(moves_data)
        assert result == 0.0

    def test_acpl_with_losses(self):
        """Test ACPL con pérdidas de centipeones."""
        moves_data = [
            {"cp_loss": 10},
            {"cp_loss": 20},
            {"cp_loss": 30}
        ]
        result = calculate_acpl(moves_data)
        assert result == 20.0

    def test_acpl_empty_moves(self):
        """Test ACPL con lista vacía."""
        result = calculate_acpl([])
        assert result == 0.0

    def test_acpl_filters_outliers(self):
        """Test que ACPL filtra outliers extremos."""
        moves_data = [
            {"cp_loss": 10},
            {"cp_loss": 500},  # Outlier extremo
            {"cp_loss": 20}
        ]
        # Debería filtrar el outlier de 500
        result = calculate_acpl(moves_data)
        assert result < 200  # Mucho menor que incluir el outlier


class TestMatchRate:
    """Tests para cálculo de match rate."""

    def test_perfect_match_rate(self):
        """Test match rate con todas las jugadas coincidentes."""
        moves_data = [
            {"played": "e4", "best": "e4"},
            {"played": "Nf3", "best": "Nf3"},
            {"played": "Bb5", "best": "Bb5"}
        ]
        result = calculate_match_rate(moves_data)
        assert result == 1.0

    def test_zero_match_rate(self):
        """Test match rate sin coincidencias."""
        moves_data = [
            {"played": "e4", "best": "d4"},
            {"played": "Nf3", "best": "Nc3"},
            {"played": "Bb5", "best": "Be2"}
        ]
        result = calculate_match_rate(moves_data)
        assert result == 0.0

    def test_partial_match_rate(self):
        """Test match rate parcial."""
        moves_data = [
            {"played": "e4", "best": "e4"},     # Match
            {"played": "Nf3", "best": "Nc3"},   # No match
            {"played": "Bb5", "best": "Bb5"}    # Match
        ]
        result = calculate_match_rate(moves_data)
        assert result == pytest.approx(0.667, rel=1e-2)


class TestMoveQualityClassification:
    """Tests para clasificación de calidad de jugadas."""

    def test_classify_excellent_move(self):
        """Test clasificación de jugada excelente."""
        result = classify_move_quality(0)  # Sin pérdida
        assert result == "excellent"

    def test_classify_good_move(self):
        """Test clasificación de jugada buena."""
        result = classify_move_quality(15)  # Pérdida pequeña
        assert result == "good"

    def test_classify_inaccuracy(self):
        """Test clasificación de imprecisión."""
        result = classify_move_quality(35)  # Pérdida moderada
        assert result == "inaccuracy"

    def test_classify_mistake(self):
        """Test clasificación de error."""
        result = classify_move_quality(75)  # Pérdida considerable
        assert result == "mistake"

    def test_classify_blunder(self):
        """Test clasificación de pifia."""
        result = classify_move_quality(150)  # Pérdida grande
        assert result == "blunder"


class TestAggregateQualityFeatures:
    """Tests para agregación de features de calidad."""

    def test_aggregate_with_sample_data(self, sample_moves_data):
        """Test agregación con datos de muestra."""
        result = aggregate_quality_features(sample_moves_data)

        # Verificar que retorna diccionario con claves esperadas
        expected_keys = [
            "avg_acpl", "avg_wdl_loss", "match_rate",
            "ipr", "blunder_rate", "mistake_rate"
        ]
        for key in expected_keys:
            assert key in result
            assert isinstance(result[key], (int, float))

    def test_aggregate_empty_data(self):
        """Test agregación con datos vacíos."""
        result = aggregate_quality_features([])

        # Con datos vacíos, debería retornar valores por defecto
        assert result["avg_acpl"] == 0.0
        assert result["match_rate"] == 0.0
        assert result["blunder_rate"] == 0.0