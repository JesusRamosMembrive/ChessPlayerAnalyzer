"""
Utility Tasks Module

This module contains utility and maintenance Celery tasks including
health checks, clustering operations, and periodic maintenance tasks.
"""
from __future__ import annotations

import time

from celery.schedules import crontab
from app.analysis.clustering import recompute_and_update_clusters

from .shared_config import logger

# Get celery app instance - will be injected by main celery_app.py
celery_app = None

def register_utility_tasks(app):
    """Register utility tasks with the Celery app instance."""
    global celery_app
    celery_app = app

    # Register tasks
    app.task(name="test_worker_functionality")(test_worker_functionality)
    app.task(name="recalculate_player_clusters")(recalculate_player_clusters)

def test_worker_functionality():
    """Simple no-op task to verify worker functionality after restart."""
    logger.info("Testing worker functionality")
    time.sleep(0.1)
    return {"status": "success", "message": "Worker functionality verified"}

def recalculate_player_clusters():
    """Periodic task to recompute clustering models and update players."""
    logger.info("Starting cluster recalculation")
    try:
        recompute_and_update_clusters()
        logger.info("Cluster recalculation completed successfully")
        return {"status": "ok", "message": "Clusters recalculated successfully"}
    except Exception as e:
        logger.error(f"Error in cluster recalculation: {e}")
        return {"status": "error", "message": f"Cluster recalculation failed: {e}"}

def setup_periodic_tasks(app):
    """Setup periodic task schedules."""
    @app.on_after_configure.connect
    def setup_cluster_schedule(sender, **kwargs):
        """Configure the daily cluster recalculation schedule."""
        sender.add_periodic_task(
            crontab(hour=0, minute=0),
            recalculate_player_clusters.s(),
            name="recalculate-clusters-daily",
        )
        logger.info("Periodic task 'recalculate-clusters-daily' scheduled")