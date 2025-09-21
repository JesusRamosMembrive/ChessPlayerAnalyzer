#!/usr/bin/env python3
"""
Pruebas exhaustivas de Value Objects.
Valida todas las reglas de negocio, validaciones y comportamientos.
"""
import sys
from datetime import datetime

# Agregar path para imports
sys.path.append('.')

# Importar Value Objects directamente evitando problemas de logging
from app.domain.value_objects.metrics import (
    QualityMetrics,
    TimingMetrics,
    OpeningMetrics,
    EndgameMetrics,
    RiskAssessment,
    PerformanceMetrics,
    PhaseQuality,
    ClutchAccuracy,
    TimeComplexity,
    Benchmark,
    Segments
)


class TestQualityMetrics:
    """Pruebas exhaustivas para QualityMetrics."""

    def test_valid_quality_metrics_creation(self):
        """Test creación válida de QualityMetrics."""
        metrics = QualityMetrics(
            avg_acpl=25.5,
            avg_wdl_loss=8.2,
            robust_loss=6.8,
            avg_match_rate=75.3,
            avg_ipr=2150.0
        )

        assert metrics.avg_acpl == 25.5
        assert metrics.avg_wdl_loss == 8.2
        assert metrics.robust_loss == 6.8
        assert metrics.avg_match_rate == 75.3
        assert metrics.avg_ipr == 2150.0

    def test_quality_metrics_immutability(self):
        """Test inmutabilidad de QualityMetrics."""
        metrics = QualityMetrics(
            avg_acpl=25.5,
            avg_wdl_loss=8.2,
            robust_loss=6.8,
            avg_match_rate=75.3,
            avg_ipr=2150.0
        )

        # No se puede modificar
        try:
            metrics.avg_acpl = 30.0
            assert False, "Should not be able to modify frozen dataclass"
        except AttributeError:
            pass  # Expected

    def test_quality_metrics_negative_acpl_validation(self):
        """Test validación de ACPL negativo."""
        try:
            QualityMetrics(
                avg_acpl=-5.0,  # Inválido
                avg_wdl_loss=8.2,
                robust_loss=6.8,
                avg_match_rate=75.3,
                avg_ipr=2150.0
            )
            assert False, "Should raise ValueError for negative ACPL"
        except ValueError as e:
            assert "avg_acpl debe ser >= 0" in str(e)

    def test_quality_metrics_invalid_wdl_loss_validation(self):
        """Test validación de WDL loss fuera de rango."""
        # WDL loss < 0
        try:
            QualityMetrics(
                avg_acpl=25.5,
                avg_wdl_loss=-1.0,  # Inválido
                robust_loss=6.8,
                avg_match_rate=75.3,
                avg_ipr=2150.0
            )
            assert False, "Should raise ValueError for negative WDL loss"
        except ValueError as e:
            assert "avg_wdl_loss debe estar entre 0 y 100" in str(e)

        # WDL loss > 100
        try:
            QualityMetrics(
                avg_acpl=25.5,
                avg_wdl_loss=101.0,  # Inválido
                robust_loss=6.8,
                avg_match_rate=75.3,
                avg_ipr=2150.0
            )
            assert False, "Should raise ValueError for WDL loss > 100"
        except ValueError as e:
            assert "avg_wdl_loss debe estar entre 0 y 100" in str(e)

    def test_quality_metrics_invalid_match_rate_validation(self):
        """Test validación de match rate fuera de rango."""
        try:
            QualityMetrics(
                avg_acpl=25.5,
                avg_wdl_loss=8.2,
                robust_loss=6.8,
                avg_match_rate=150.0,  # Inválido
                avg_ipr=2150.0
            )
            assert False, "Should raise ValueError for match rate > 100"
        except ValueError as e:
            assert "avg_match_rate debe estar entre 0 y 100" in str(e)

    def test_quality_metrics_edge_cases(self):
        """Test casos límite para QualityMetrics."""
        # Valores mínimos
        metrics_min = QualityMetrics(
            avg_acpl=0.0,
            avg_wdl_loss=0.0,
            robust_loss=0.0,
            avg_match_rate=0.0,
            avg_ipr=0.0
        )
        assert metrics_min.avg_acpl == 0.0

        # Valores máximos
        metrics_max = QualityMetrics(
            avg_acpl=999.9,
            avg_wdl_loss=100.0,
            robust_loss=999.9,
            avg_match_rate=100.0,
            avg_ipr=9999.0
        )
        assert metrics_max.avg_wdl_loss == 100.0


class TestTimingMetrics:
    """Pruebas exhaustivas para TimingMetrics."""

    def test_valid_timing_metrics_creation(self):
        """Test creación válida de TimingMetrics."""
        metrics = TimingMetrics(
            mean_move_time=15.3,
            time_variance=45.2,
            uniformity_score=0.85,
            lag_spike_count=3
        )

        assert metrics.mean_move_time == 15.3
        assert metrics.time_variance == 45.2
        assert metrics.uniformity_score == 0.85
        assert metrics.lag_spike_count == 3

    def test_timing_metrics_negative_validation(self):
        """Test validación de valores negativos en TimingMetrics."""
        try:
            TimingMetrics(
                mean_move_time=-5.0,  # Inválido
                time_variance=45.2,
                uniformity_score=0.85,
                lag_spike_count=3
            )
            assert False, "Should raise ValueError for negative mean_move_time"
        except ValueError as e:
            assert "mean_move_time debe ser >= 0" in str(e)

    def test_timing_metrics_uniformity_score_range(self):
        """Test validación de rango de uniformity_score."""
        try:
            TimingMetrics(
                mean_move_time=15.3,
                time_variance=45.2,
                uniformity_score=1.5,  # Inválido > 1
                lag_spike_count=3
            )
            assert False, "Should raise ValueError for uniformity_score > 1"
        except ValueError as e:
            assert "uniformity_score debe estar entre 0 y 1" in str(e)


class TestRiskAssessment:
    """Pruebas exhaustivas para RiskAssessment."""

    def test_valid_risk_assessment_creation(self):
        """Test creación válida de RiskAssessment."""
        assessment = RiskAssessment(
            risk_score=75,
            risk_factors={"high_match_rate": True, "low_acpl": False},
            confidence_level=85
        )

        assert assessment.risk_score == 75
        assert assessment.risk_factors["high_match_rate"] is True
        assert assessment.risk_factors["low_acpl"] is False
        assert assessment.confidence_level == 85

    def test_risk_assessment_score_validation(self):
        """Test validación de risk_score."""
        try:
            RiskAssessment(
                risk_score=150,  # Inválido > 100
                risk_factors={},
                confidence_level=85
            )
            assert False, "Should raise ValueError for risk_score > 100"
        except ValueError as e:
            assert "risk_score debe estar entre 0 y 100" in str(e)

    def test_risk_assessment_confidence_validation(self):
        """Test validación de confidence_level."""
        try:
            RiskAssessment(
                risk_score=75,
                risk_factors={},
                confidence_level=-10  # Inválido < 0
            )
            assert False, "Should raise ValueError for negative confidence_level"
        except ValueError as e:
            assert "confidence_level debe estar entre 0 y 100" in str(e)

    def test_risk_assessment_risk_factors_type(self):
        """Test validación del tipo de risk_factors."""
        try:
            RiskAssessment(
                risk_score=75,
                risk_factors="not_a_dict",  # Inválido
                confidence_level=85
            )
            assert False, "Should raise ValueError for non-dict risk_factors"
        except ValueError as e:
            assert "risk_factors debe ser un diccionario" in str(e)


class TestOpeningMetrics:
    """Pruebas exhaustivas para OpeningMetrics."""

    def test_valid_opening_metrics_creation(self):
        """Test creación válida de OpeningMetrics."""
        metrics = OpeningMetrics(
            book_depth=8,
            mean_entropy=0.75,
            novelty_rate=0.15,
            theory_adherence=0.90
        )

        assert metrics.book_depth == 8
        assert metrics.mean_entropy == 0.75
        assert metrics.novelty_rate == 0.15
        assert metrics.theory_adherence == 0.90

    def test_opening_metrics_book_depth_validation(self):
        """Test validación de book_depth."""
        try:
            OpeningMetrics(
                book_depth=-1,  # Inválido
                mean_entropy=0.75,
                novelty_rate=0.15,
                theory_adherence=0.90
            )
            assert False, "Should raise ValueError for negative book_depth"
        except ValueError as e:
            assert "book_depth debe ser >= 0" in str(e)

    def test_opening_metrics_entropy_range_validation(self):
        """Test validación de rango de mean_entropy."""
        try:
            OpeningMetrics(
                book_depth=8,
                mean_entropy=1.5,  # Inválido > 1
                novelty_rate=0.15,
                theory_adherence=0.90
            )
            assert False, "Should raise ValueError for mean_entropy > 1"
        except ValueError as e:
            assert "mean_entropy debe estar entre 0 y 1" in str(e)


class TestEndgameMetrics:
    """Pruebas exhaustivas para EndgameMetrics."""

    def test_valid_endgame_metrics_creation(self):
        """Test creación válida de EndgameMetrics."""
        metrics = EndgameMetrics(
            conversion_rate=0.85,
            defensive_accuracy=0.78,
            technique_score=0.82,
            piece_activity=0.90
        )

        assert metrics.conversion_rate == 0.85
        assert metrics.defensive_accuracy == 0.78
        assert metrics.technique_score == 0.82
        assert metrics.piece_activity == 0.90

    def test_endgame_metrics_range_validation(self):
        """Test validación de rangos en EndgameMetrics."""
        try:
            EndgameMetrics(
                conversion_rate=1.5,  # Inválido > 1
                defensive_accuracy=0.78,
                technique_score=0.82,
                piece_activity=0.90
            )
            assert False, "Should raise ValueError for conversion_rate > 1"
        except ValueError as e:
            assert "conversion_rate debe estar entre 0 y 1" in str(e)


class TestComplexValueObjects:
    """Pruebas para Value Objects más complejos."""

    def test_performance_metrics_creation(self):
        """Test creación de PerformanceMetrics."""
        metrics = PerformanceMetrics(
            elo_estimate=2150,
            performance_rating=2200,
            roi_curve=[1.0, 1.1, 1.05, 1.08],
            strength_variance=50.5
        )

        assert metrics.elo_estimate == 2150
        assert metrics.performance_rating == 2200
        assert len(metrics.roi_curve) == 4
        assert metrics.strength_variance == 50.5

    def test_phase_quality_creation(self):
        """Test creación de PhaseQuality."""
        phase = PhaseQuality(
            opening_acpl=20.5,
            middlegame_acpl=30.2,
            endgame_acpl=18.7,
            opening_blunder_rate=0.02,
            middlegame_blunder_rate=0.05,
            endgame_blunder_rate=0.03,
            blunder_rate=0.033
        )

        assert phase.opening_acpl == 20.5
        assert phase.middlegame_acpl == 30.2
        assert phase.endgame_acpl == 18.7
        assert phase.blunder_rate == 0.033

    def test_benchmark_creation(self):
        """Test creación de Benchmark."""
        benchmark = Benchmark(
            percentile_acpl=75,
            percentile_entropy=68
        )

        assert benchmark.percentile_acpl == 75
        assert benchmark.percentile_entropy == 68

    def test_segments_creation(self):
        """Test creación de Segments."""
        segment = Segments(
            start=0,
            end=50,
            mean_acpl=25.5,
            mean_time=15.3
        )

        assert segment.start == 0
        assert segment.end == 50
        assert segment.mean_acpl == 25.5
        assert segment.mean_time == 15.3

    def test_segments_validation(self):
        """Test validación de Segments."""
        try:
            Segments(
                start=50,  # start > end - inválido
                end=25,
                mean_acpl=25.5,
                mean_time=15.3
            )
            assert False, "Should raise ValueError for start > end"
        except ValueError as e:
            assert "start debe ser <= end" in str(e)


def run_value_objects_tests():
    """Ejecutar todas las pruebas de Value Objects."""
    print("🧪 Running Exhaustive Value Objects Tests...")

    # QualityMetrics tests
    print("  📝 Testing QualityMetrics...")
    test_quality = TestQualityMetrics()
    test_quality.test_valid_quality_metrics_creation()
    test_quality.test_quality_metrics_immutability()
    test_quality.test_quality_metrics_negative_acpl_validation()
    test_quality.test_quality_metrics_invalid_wdl_loss_validation()
    test_quality.test_quality_metrics_invalid_match_rate_validation()
    test_quality.test_quality_metrics_edge_cases()
    print("    ✅ QualityMetrics tests passed")

    # TimingMetrics tests
    print("  📝 Testing TimingMetrics...")
    test_timing = TestTimingMetrics()
    test_timing.test_valid_timing_metrics_creation()
    test_timing.test_timing_metrics_negative_validation()
    test_timing.test_timing_metrics_uniformity_score_range()
    print("    ✅ TimingMetrics tests passed")

    # RiskAssessment tests
    print("  📝 Testing RiskAssessment...")
    test_risk = TestRiskAssessment()
    test_risk.test_valid_risk_assessment_creation()
    test_risk.test_risk_assessment_score_validation()
    test_risk.test_risk_assessment_confidence_validation()
    test_risk.test_risk_assessment_risk_factors_type()
    print("    ✅ RiskAssessment tests passed")

    # OpeningMetrics tests
    print("  📝 Testing OpeningMetrics...")
    test_opening = TestOpeningMetrics()
    test_opening.test_valid_opening_metrics_creation()
    test_opening.test_opening_metrics_book_depth_validation()
    test_opening.test_opening_metrics_entropy_range_validation()
    print("    ✅ OpeningMetrics tests passed")

    # EndgameMetrics tests
    print("  📝 Testing EndgameMetrics...")
    test_endgame = TestEndgameMetrics()
    test_endgame.test_valid_endgame_metrics_creation()
    test_endgame.test_endgame_metrics_range_validation()
    print("    ✅ EndgameMetrics tests passed")

    # Complex Value Objects tests
    print("  📝 Testing Complex Value Objects...")
    test_complex = TestComplexValueObjects()
    test_complex.test_performance_metrics_creation()
    test_complex.test_phase_quality_creation()
    test_complex.test_benchmark_creation()
    test_complex.test_segments_creation()
    test_complex.test_segments_validation()
    print("    ✅ Complex Value Objects tests passed")

    print("✅ All Value Objects tests PASSED!")
    return True


if __name__ == "__main__":
    try:
        success = run_value_objects_tests()
        print("\n🎉 Value Objects exhaustive testing completed successfully!")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Value Objects tests failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)