"""Chess Player Analyzer - Modular Celery Application

This module provides the main Celery application configuration and
imports modular task definitions from specialized task modules.

The tasks have been organized into the following modules:
- analysis_tasks: Game and player analysis tasks
- workflow_tasks: Orchestration and workflow tasks
- utility_tasks: Maintenance and utility tasks
"""

from __future__ import annotations

import os
import logging

from celery import Celery
from kombu import Queue

# Import task registration functions
from app.tasks import (
    register_analysis_tasks,
    register_workflow_tasks,
    register_utility_tasks,
    setup_periodic_tasks
)

logger = logging.getLogger(__name__)


# Celery Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
celery_app = Celery("chess_tasks", broker=REDIS_URL, backend=REDIS_URL)

# ──────────────────────────────────────────────────────────────
#  Task timeout & retry configuration (env-driven)
# ──────────────────────────────────────────────────────────────
TASK_SOFT_TIME_LIMIT = int(os.getenv("TASK_SOFT_TIME_LIMIT", "1800"))  # 30 min default
TASK_TIME_LIMIT      = int(os.getenv("TASK_TIME_LIMIT", "1860"))      # hard limit (soft + 1 min)
TASK_MAX_RETRIES     = int(os.getenv("TASK_MAX_RETRIES", "3"))         # default max retries

# Declarar la cola por defecto con soporte de prioridad (máx. 10 en Redis)
celery_app.conf.task_default_queue = "default"
# El tuple final debe contener únicamente el objeto Queue
celery_app.conf.task_queues = (
    Queue("default", max_priority=10),
    Queue("torch", max_priority=10),
)

celery_app.conf.update(
    task_reject_on_worker_lost=True,
    task_acks_late=True,
    task_soft_time_limit=TASK_SOFT_TIME_LIMIT,
    task_time_limit=TASK_TIME_LIMIT,
    task_default_retry_delay=60,
    task_default_max_retries=TASK_MAX_RETRIES,
)
celery_app.conf.task_routes = {
    "app.ml_tasks.*": {"queue": "torch"},
}

# ──────────────────────────────────────────────────────────────
#  Register tasks from modular task modules
# ──────────────────────────────────────────────────────────────
logger.info("Registering modular Celery tasks...")

# Register tasks from each module
register_analysis_tasks(celery_app)
register_workflow_tasks(celery_app)
register_utility_tasks(celery_app)

# Setup periodic tasks
setup_periodic_tasks(celery_app)

logger.info("Modular Celery tasks registered successfully")

# ──────────────────────────────────────────────────────────────
# Import task access points for backward compatibility
# ──────────────────────────────────────────────────────────────

# Import individual tasks for external use
from app.tasks.analysis_tasks import (
    analyze_player_detailed,
    analyze_game_task,
    analyze_game_detailed,
    extract_game_id
)
from app.tasks.workflow_tasks import (
    process_player_enhanced
)
from app.tasks.utility_tasks import (
    test_worker_functionality,
    recalculate_player_clusters
)

# Make tasks available at module level for backward compatibility
__all__ = [
    'celery_app',
    'analyze_player_detailed',
    'analyze_game_task',
    'analyze_game_detailed',
    'extract_game_id',
    'process_player_enhanced',
    'test_worker_functionality',
    'recalculate_player_clusters'
]
