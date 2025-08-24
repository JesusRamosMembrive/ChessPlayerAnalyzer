"""
Módulo de análisis avanzado de partidas de ajedrez.
"""

# Hacer disponibles los módulos principales
from . import quality
from . import timing
from . import openings
from . import endgame
from . import longitudinal
from . import spc
from .engine import ChessAnalysisEngine, prepare_moves_dataframe

from .quality   import aggregate_quality_features
from .timing    import aggregate_time_features
from .openings  import aggregate_opening_features
from .endgame   import aggregate_endgame_features
from .spc       import compute_spc

__all__ = [
    'quality',
    'timing', 
    'openings',
    'endgame',
    'longitudinal',
    'spc',
    'prepare_moves_dataframe',
    'ChessAnalysisEngine',
    'aggregate_quality_features',
    'aggregate_time_features',
    'aggregate_opening_features',
    'aggregate_endgame_features',
    'compute_spc',
]
