"""
Módulo de análisis avanzado de partidas de ajedrez.
"""

# Hacer disponibles los módulos principales
from . import quality
from . import timing
from . import openings
from . import endgame
from . import longitudinal
# BULLDOZER TOTAL: Use bulldozer_engine instead
from .bulldozer_engine import analyze_game_complete as AnalysisEngine

from .quality   import aggregate_quality_features
from .timing    import aggregate_time_features
from .openings  import aggregate_opening_features
from .endgame   import aggregate_endgame_features

__all__ = [
    'quality',
    'timing',
    'openings',
    'endgame',
    'longitudinal',
    'AnalysisEngine',
    'aggregate_quality_features',
    'aggregate_time_features',
    'aggregate_opening_features',
    'aggregate_endgame_features',
]
