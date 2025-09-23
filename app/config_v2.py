# app/config_v2.py
"""
Configuración para el refactor V2 - Flag de alternancia entre versiones
"""
import os
from typing import Literal


class RefactorConfig:
    """Configuración para el refactor V2"""

    # Flag principal para alternar entre V1 y V2
    USE_V2_ENGINE: bool = os.getenv("USE_V2_ENGINE", "false").lower() == "true"

    # Flags específicos por funcionalidad
    USE_V2_MODELS: bool = os.getenv("USE_V2_MODELS", "false").lower() == "true"
    USE_V2_TASKS: bool = os.getenv("USE_V2_TASKS", "false").lower() == "true"
    USE_V2_ANALYSIS: bool = os.getenv("USE_V2_ANALYSIS", "false").lower() == "true"

    # Modo híbrido: usar V2 solo para nuevos análisis
    HYBRID_MODE: bool = os.getenv("HYBRID_MODE", "false").lower() == "true"

    # Debug: logging adicional para V2
    DEBUG_V2: bool = os.getenv("DEBUG_V2", "false").lower() == "true"

    @classmethod
    def get_engine_version(cls) -> Literal["v1", "v2"]:
        """Retorna la versión del motor a usar"""
        return "v2" if cls.USE_V2_ENGINE else "v1"

    @classmethod
    def should_use_v2_for_player(cls, username: str) -> bool:
        """Determina si usar V2 para un jugador específico"""
        if cls.USE_V2_ENGINE:
            return True

        if cls.HYBRID_MODE:
            # En modo híbrido, usar V2 para jugadores nuevos o específicos
            # Puedes agregar lógica aquí para determinar qué jugadores usar con V2
            return False

        return False

    @classmethod
    def log_config(cls):
        """Log de configuración actual"""
        import logging
        logger = logging.getLogger(__name__)

        logger.info(f"Refactor V2 Configuration:")
        logger.info(f"  USE_V2_ENGINE: {cls.USE_V2_ENGINE}")
        logger.info(f"  USE_V2_MODELS: {cls.USE_V2_MODELS}")
        logger.info(f"  USE_V2_TASKS: {cls.USE_V2_TASKS}")
        logger.info(f"  USE_V2_ANALYSIS: {cls.USE_V2_ANALYSIS}")
        logger.info(f"  HYBRID_MODE: {cls.HYBRID_MODE}")
        logger.info(f"  DEBUG_V2: {cls.DEBUG_V2}")
        logger.info(f"  Engine version: {cls.get_engine_version()}")


# Instancia global de configuración
config_v2 = RefactorConfig()


# Environment variables documentation:
"""
Variables de entorno para configurar el refactor V2:

USE_V2_ENGINE=true|false
    - true: Usar completamente el motor V2 (recomendado para testing)
    - false: Usar motor V1 (default, producción actual)

USE_V2_MODELS=true|false
    - true: Usar modelos V2 (Game, AnalysisResult, Player)
    - false: Usar modelos V1 (default)

USE_V2_TASKS=true|false
    - true: Usar celery_tasks_v2.py
    - false: Usar celery_app.py (default)

USE_V2_ANALYSIS=true|false
    - true: Usar AnalysisEngine V2
    - false: Usar análisis original (default)

HYBRID_MODE=true|false
    - true: Modo híbrido - usar V2 solo para casos específicos
    - false: Usar solo la versión configurada (default)

DEBUG_V2=true|false
    - true: Logging adicional para debugging V2
    - false: Logging normal (default)

Ejemplos de uso:

# Testing completo V2
USE_V2_ENGINE=true USE_V2_MODELS=true USE_V2_TASKS=true USE_V2_ANALYSIS=true

# Producción actual (sin cambios)
# (no configurar nada, usa defaults)

# Testing gradual
HYBRID_MODE=true USE_V2_TASKS=true

# Debug intensivo
USE_V2_ENGINE=true DEBUG_V2=true
"""