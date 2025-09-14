"""
Modular Celery Tasks Package

This package organizes Celery tasks into specialized modules for better maintainability:

- analysis_tasks: Game and player analysis tasks (Stockfish, detailed metrics)
- workflow_tasks: Orchestration and workflow coordination tasks
- utility_tasks: Maintenance, health checks, and periodic tasks
- shared_config: Common configuration and utilities
"""

from .analysis_tasks import register_analysis_tasks
from .workflow_tasks import register_workflow_tasks
from .utility_tasks import register_utility_tasks, setup_periodic_tasks

__all__ = [
    'register_analysis_tasks',
    'register_workflow_tasks',
    'register_utility_tasks',
    'setup_periodic_tasks'
]