# app/adapters/analysis_adapter.py
"""
Adaptador que permite alternar entre V1 y V2 según configuración.
Mantiene la misma interfaz para los endpoints.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from sqlmodel import Session, select

# Configuración
from app.config_v2 import config_v2

# Modelos V1 y V2
from app import models as models_v1
from app.models_v2 import Player as PlayerV2, Game as GameV2, AnalysisResult as AnalysisResultV2

# Tasks V1 y V2
from app.celery_app import process_player_enhanced as process_player_v1
from app.celery_tasks_v2 import (
    process_player_enhanced_v2,
    analyze_player_v2,
    celery_app_v2
)

# Database
from app.database import engine

logger = logging.getLogger(__name__)


class AnalysisAdapter:
    """
    Adaptador que mantiene la misma interfaz pero usa V1 o V2 según configuración.
    """

    def __init__(self):
        self.version = config_v2.get_engine_version()
        if config_v2.DEBUG_V2:
            logger.info(f"AnalysisAdapter initialized with version: {self.version}")

    def get_player_status(self, username: str, session: Session) -> Dict[str, Any]:
        """
        Obtiene el estado de un jugador, compatible con endpoint GET /players/{username}
        """
        if config_v2.DEBUG_V2:
            logger.info(f"Getting player status for {username} using {self.version}")

        if self.version == "v2" or config_v2.should_use_v2_for_player(username):
            return self._get_player_status_v2(username, session)
        else:
            return self._get_player_status_v1(username, session)

    def _get_player_status_v1(self, username: str, session: Session) -> Dict[str, Any]:
        """Obtiene estado usando modelos V1"""
        player = session.get(models_v1.Player, username)

        if not player:
            return {
                "username": username,
                "status": "not_analyzed",
                "progress": 0,
                "total_games": 0,
                "done_games": 0,
                "requested_at": None,
                "finished_at": None,
                "error": None,
                "last_task_id": None
            }

        return {
            "username": player.username,
            "status": player.status.value if hasattr(player.status, 'value') else player.status,
            "progress": player.progress,
            "total_games": player.total_games,
            "done_games": player.done_games,
            "requested_at": player.requested_at.isoformat() if player.requested_at else None,
            "finished_at": player.finished_at.isoformat() if player.finished_at else None,
            "error": player.error,
            "last_task_id": player.last_task_id
        }

    def _get_player_status_v2(self, username: str, session: Session) -> Dict[str, Any]:
        """Obtiene estado usando modelos V2"""
        player = session.exec(
            select(PlayerV2).where(PlayerV2.username == username)
        ).first()

        if not player:
            return {
                "username": username,
                "status": "not_analyzed",
                "progress": 0,
                "total_games": 0,
                "done_games": 0,
                "requested_at": None,
                "finished_at": None,
                "error": None,
                "last_task_id": None
            }

        return {
            "username": player.username,
            "status": player.status.value if hasattr(player.status, 'value') else player.status,
            "progress": player.progress,
            "total_games": player.total_games,
            "done_games": player.done_games,
            "requested_at": player.requested_at.isoformat() if player.requested_at else None,
            "finished_at": player.finished_at.isoformat() if player.finished_at else None,
            "error": player.error,
            "last_task_id": player.last_task_id
        }

    def start_player_analysis(self, username: str, force_reanalysis: bool = False) -> Dict[str, Any]:
        """
        Inicia análisis de jugador, compatible con endpoint POST /players/{username}
        """
        if config_v2.DEBUG_V2:
            logger.info(f"Starting player analysis for {username} using {self.version}")

        if self.version == "v2" or config_v2.should_use_v2_for_player(username):
            return self._start_player_analysis_v2(username, force_reanalysis)
        else:
            return self._start_player_analysis_v1(username, force_reanalysis)

    def _start_player_analysis_v1(self, username: str, force_reanalysis: bool) -> Dict[str, Any]:
        """Inicia análisis usando tasks V1"""
        task = process_player_v1.delay(username, force_reanalysis)

        # Actualizar player en V1
        with Session(engine) as session:
            player = session.get(models_v1.Player, username)
            if not player:
                player = models_v1.Player(
                    username=username,
                    status=models_v1.PlayerStatus.pending,
                    requested_at=datetime.now(timezone.utc),
                    last_task_id=task.id
                )
                session.add(player)
            else:
                player.status = models_v1.PlayerStatus.pending
                player.last_task_id = task.id
                player.requested_at = datetime.now(timezone.utc)
                session.add(player)
            session.commit()

        return {
            "message": f"Analysis started for player {username}",
            "task_id": task.id,
            "username": username,
            "status": "pending"
        }

    def _start_player_analysis_v2(self, username: str, force_reanalysis: bool) -> Dict[str, Any]:
        """Inicia análisis usando tasks V2"""
        task = process_player_enhanced_v2.delay(username, force_reanalysis)

        # Actualizar player en V2
        with Session(engine) as session:
            player = session.exec(
                select(PlayerV2).where(PlayerV2.username == username)
            ).first()

            if not player:
                player = PlayerV2(
                    username=username,
                    status="pending",
                    requested_at=datetime.now(timezone.utc),
                    last_task_id=task.id
                )
                session.add(player)
            else:
                player.status = "pending"
                player.last_task_id = task.id
                player.requested_at = datetime.now(timezone.utc)
                session.add(player)
            session.commit()

        return {
            "message": f"Analysis started for player {username}",
            "task_id": task.id,
            "username": username,
            "status": "pending"
        }

    def get_player_metrics(self, username: str, session: Session) -> Dict[str, Any]:
        """
        Obtiene métricas de jugador, compatible con endpoint GET /metrics/player/{username}
        """
        if config_v2.DEBUG_V2:
            logger.info(f"Getting player metrics for {username} using {self.version}")

        if self.version == "v2" or config_v2.should_use_v2_for_player(username):
            return self._get_player_metrics_v2(username, session)
        else:
            return self._get_player_metrics_v1(username, session)

    def _get_player_metrics_v1(self, username: str, session: Session) -> Dict[str, Any]:
        """Obtiene métricas usando modelos V1"""
        obj = session.get(models_v1.PlayerAnalysisDetailed, username)
        if not obj:
            raise ValueError("No metrics yet")

        # Convertir a formato compatible (código original de main.py)
        # Este es el formato que espera React
        response_data = {
            'username': obj.username,
            'games_analyzed': obj.games_analyzed,
            'avg_acpl': obj.avg_acpl,
            'avg_wdl_loss': obj.avg_wdl_loss,
            'robust_loss': obj.robust_loss,
            'std_acpl': obj.std_acpl,
            'avg_match_rate': obj.avg_match_rate,
            'std_match_rate': obj.std_match_rate,
            'avg_ipr': obj.avg_ipr,
            # ... resto de campos según main.py original
        }

        return response_data

    def _get_player_metrics_v2(self, username: str, session: Session) -> Dict[str, Any]:
        """Obtiene métricas usando modelos V2"""
        player = session.exec(
            select(PlayerV2).where(PlayerV2.username == username)
        ).first()

        if not player or not player.aggregated_metrics:
            raise ValueError("No metrics yet")

        # Las métricas ya están en el formato correcto para React
        return player.aggregated_metrics

    def delete_player(self, username: str, session: Session) -> bool:
        """Elimina un jugador según la versión"""
        if self.version == "v2" or config_v2.should_use_v2_for_player(username):
            player = session.exec(
                select(PlayerV2).where(PlayerV2.username == username)
            ).first()
            if player:
                session.delete(player)
                session.commit()
                return True
        else:
            player = session.get(models_v1.Player, username)
            if player:
                session.delete(player)
                session.commit()
                return True

        return False

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Obtiene estado de task según la versión"""
        if self.version == "v2":
            try:
                from app.celery_tasks_v2 import get_task_result_v2
                return get_task_result_v2(task_id)
            except ImportError:
                # Fallback a V1
                pass

        # V1 o fallback
        from app.celery_app import celery_app
        result = celery_app.AsyncResult(task_id)
        return {
            'task_id': task_id,
            'status': result.status,
            'result': result.result,
            'ready': result.ready()
        }


# Instancia global del adaptador
analysis_adapter = AnalysisAdapter()