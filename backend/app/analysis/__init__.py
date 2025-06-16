"""
Módulo de análisis avanzado de partidas de ajedrez.
"""

# Hacer disponibles los módulos principales
from . import quality
from . import timing
from . import openings
from . import endgame
from . import longitudinal
from .engine import ChessAnalysisEngine

__all__ = [
    'quality',
    'timing', 
    'openings',
    'endgame',
    'longitudinal',
    'ChessAnalysisEngine'
]
