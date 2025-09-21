"""
Tests unitarios para value objects.
"""
import pytest
from app.domain.value_objects.metrics import (
    QualityMetrics,
    TimingMetrics,
    OpeningMetrics,
    RiskAssessment
)


class TestQualityMetrics:
    """Tests para QualityMetrics value object."""

    def test_create_valid_quality_metrics(self):
        """Test creación de métricas válidas."""
        metrics = QualityMetrics(
            avg_acpl=15.5,
            avg_wdl_loss=0.05,
            robust_loss=0.02,
            avg_match_rate=0.75,
            avg_ipr=2200.0
        )
        assert metrics.avg_acpl == 15.5
        assert metrics.avg_match_rate == 0.75

    def test_quality_metrics_validation_negative_acpl(self):
        """Test validación de ACPL negativo."""
        with pytest.raises(ValueError, match="avg_acpl debe ser >= 0"):
            QualityMetrics(
                avg_acpl=-5.0,
                avg_wdl_loss=0.05,
                robust_loss=0.02,
                avg_match_rate=0.75,
                avg_ipr=2200.0
            )

    def test_quality_metrics_validation_invalid_match_rate(self):
        """Test validación de match rate fuera de rango."""
        with pytest.raises(ValueError, match="avg_match_rate debe estar entre 0 y 1"):
            QualityMetrics(
                avg_acpl=15.5,
                avg_wdl_loss=0.05,
                robust_loss=0.02,
                avg_match_rate=1.5,  # Invalid > 1
                avg_ipr=2200.0
            )

    def test_quality_metrics_immutable(self):
        """Test que QualityMetrics es inmutable."""
        metrics = QualityMetrics(
            avg_acpl=15.5,
            avg_wdl_loss=0.05,
            robust_loss=0.02,
            avg_match_rate=0.75,
            avg_ipr=2200.0
        )

        # Intentar modificar debe fallar
        with pytest.raises(AttributeError):
            metrics.avg_acpl = 20.0


class TestTimingMetrics:
    """Tests para TimingMetrics value object."""

    def test_create_valid_timing_metrics(self):
        """Test creación de métricas de timing válidas."""
        metrics = TimingMetrics(
            mean_move_time=45.2,
            time_variance=123.5,
            uniformity_score=0.8,
            lag_spike_count=5
        )
        assert metrics.mean_move_time == 45.2
        assert metrics.lag_spike_count == 5

    def test_timing_metrics_validation_negative_time(self):
        """Test validación de tiempo negativo."""
        with pytest.raises(ValueError, match="mean_move_time debe ser >= 0"):
            TimingMetrics(
                mean_move_time=-10.0,
                time_variance=123.5,
                uniformity_score=0.8,
                lag_spike_count=5
            )


class TestRiskAssessment:
    """Tests para RiskAssessment value object."""

    def test_create_valid_risk_assessment(self):
        """Test creación de evaluación de riesgo válida."""
        risk = RiskAssessment(
            risk_score=75,
            risk_factors={"low_acpl": True, "high_roi": False},
            confidence_level=85
        )
        assert risk.risk_score == 75
        assert risk.risk_factors["low_acpl"] is True

    def test_risk_assessment_validation_invalid_score(self):
        """Test validación de score fuera de rango."""
        with pytest.raises(ValueError, match="risk_score debe estar entre 0 y 100"):
            RiskAssessment(
                risk_score=150,  # Invalid > 100
                risk_factors={},
                confidence_level=85
            )