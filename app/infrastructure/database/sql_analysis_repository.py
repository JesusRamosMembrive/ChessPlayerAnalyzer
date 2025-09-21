"""
Implementación SQL del repositorio de análisis.
"""
from typing import Optional, List
from sqlmodel import Session, select, or_

from app.domain.entities.analysis import PlayerAnalysis as DomainPlayerAnalysis, GameAnalysis as DomainGameAnalysis
from app.domain.repositories.analysis_repository import AnalysisRepository
from app.models import PlayerAnalysisDetailed as SQLPlayerAnalysis
from app.models import GameAnalysisDetailed as SQLGameAnalysis, Game as SQLGame
from .mappers import analysis_to_domain, domain_to_analysis


class SQLAnalysisRepository(AnalysisRepository):
    """
    Implementación SQL del repositorio de análisis.
    """

    def __init__(self, session: Session):
        self.session = session

    async def get_player_analysis(self, username: str) -> Optional[DomainPlayerAnalysis]:
        """Obtiene el análisis de un jugador."""
        sql_analysis = self.session.get(SQLPlayerAnalysis, username)
        if sql_analysis:
            return analysis_to_domain(sql_analysis)
        return None

    async def save_player_analysis(self, analysis: DomainPlayerAnalysis) -> DomainPlayerAnalysis:
        """Guarda análisis de jugador."""
        # Verificar si ya existe
        existing = self.session.get(SQLPlayerAnalysis, analysis.username)

        if existing:
            # Actualizar existente - convertir domain a SQL y copiar campos
            updated_sql = domain_to_analysis(analysis)

            # Copiar todos los campos (excepto username que es PK)
            existing.games_analyzed = updated_sql.games_analyzed
            existing.avg_acpl = updated_sql.avg_acpl
            existing.avg_wdl_loss = updated_sql.avg_wdl_loss
            existing.robust_loss = updated_sql.robust_loss
            existing.avg_match_rate = updated_sql.avg_match_rate
            existing.avg_ipr = updated_sql.avg_ipr
            existing.std_acpl = updated_sql.std_acpl
            existing.std_match_rate = updated_sql.std_match_rate
            existing.roi_mean = updated_sql.roi_mean
            existing.roi_max = updated_sql.roi_max
            existing.roi_std = updated_sql.roi_std
            existing.step_function_detected = updated_sql.step_function_detected
            existing.step_function_magnitude = updated_sql.step_function_magnitude
            existing.peer_delta_acpl = updated_sql.peer_delta_acpl
            existing.peer_delta_match = updated_sql.peer_delta_match
            existing.longest_streak = updated_sql.longest_streak
            existing.selectivity_score = updated_sql.selectivity_score
            existing.time_management = updated_sql.time_management
            existing.opening_patterns = updated_sql.opening_patterns
            existing.performance = updated_sql.performance
            existing.phase_quality = updated_sql.phase_quality
            existing.clutch_accuracy = updated_sql.clutch_accuracy
            existing.endgame = updated_sql.endgame
            existing.time_complexity = updated_sql.time_complexity
            existing.benchmark = updated_sql.benchmark
            existing.risk_score = updated_sql.risk_score
            existing.risk_factors = updated_sql.risk_factors
            existing.confidence_level = updated_sql.confidence_level
            existing.first_game_date = updated_sql.first_game_date
            existing.last_game_date = updated_sql.last_game_date
            existing.analyzed_at = updated_sql.analyzed_at
            existing.suspicious_games_ids = updated_sql.suspicious_games_ids
            existing.tactical = updated_sql.tactical
            existing.time_patterns = updated_sql.time_patterns

            sql_analysis = existing
        else:
            # Crear nuevo
            sql_analysis = domain_to_analysis(analysis)
            self.session.add(sql_analysis)

        self.session.commit()
        self.session.refresh(sql_analysis)

        return analysis_to_domain(sql_analysis)

    async def get_game_analysis(self, game_id: int) -> Optional[DomainGameAnalysis]:
        """Obtiene el análisis de una partida."""
        sql_analysis = self.session.get(SQLGameAnalysis, game_id)
        if sql_analysis:
            # Nota: Aquí necesitaríamos un mapper específico para GameAnalysis
            # Por ahora retornamos None hasta implementar
            return None
        return None

    async def save_game_analysis(self, analysis: DomainGameAnalysis) -> DomainGameAnalysis:
        """Guarda análisis de partida."""
        # Nota: Similar al anterior, necesitaríamos mapper específico
        # Por ahora implementación simplificada
        return analysis

    async def get_game_analyses_by_player(self, username: str) -> List[DomainGameAnalysis]:
        """Obtiene todos los análisis de partidas de un jugador."""
        # Obtener todas las partidas del jugador que tengan análisis
        statement = select(SQLGameAnalysis).join(SQLGame).where(
            or_(
                SQLGame.white_username == username,
                SQLGame.black_username == username
            )
        )

        sql_analyses = self.session.exec(statement).all()

        # Por ahora retornamos lista vacía hasta implementar mapper completo
        return []

    async def delete_player_analysis(self, username: str) -> bool:
        """Elimina análisis de un jugador."""
        sql_analysis = self.session.get(SQLPlayerAnalysis, username)
        if sql_analysis:
            self.session.delete(sql_analysis)
            self.session.commit()
            return True
        return False

    # Métodos adicionales específicos de SQL

    def get_player_analysis_sync(self, username: str) -> Optional[DomainPlayerAnalysis]:
        """Versión síncrona para compatibilidad."""
        sql_analysis = self.session.get(SQLPlayerAnalysis, username)
        if sql_analysis:
            return analysis_to_domain(sql_analysis)
        return None

    def save_player_analysis_sync(self, analysis: DomainPlayerAnalysis) -> DomainPlayerAnalysis:
        """Versión síncrona para compatibilidad."""
        existing = self.session.get(SQLPlayerAnalysis, analysis.username)

        if existing:
            updated_sql = domain_to_analysis(analysis)

            # Copiar campos (lógica duplicada para sincronía)
            existing.games_analyzed = updated_sql.games_analyzed
            existing.avg_acpl = updated_sql.avg_acpl
            existing.avg_wdl_loss = updated_sql.avg_wdl_loss
            existing.robust_loss = updated_sql.robust_loss
            existing.avg_match_rate = updated_sql.avg_match_rate
            existing.avg_ipr = updated_sql.avg_ipr
            existing.std_acpl = updated_sql.std_acpl
            existing.std_match_rate = updated_sql.std_match_rate
            existing.roi_mean = updated_sql.roi_mean
            existing.roi_max = updated_sql.roi_max
            existing.roi_std = updated_sql.roi_std
            existing.step_function_detected = updated_sql.step_function_detected
            existing.step_function_magnitude = updated_sql.step_function_magnitude
            existing.peer_delta_acpl = updated_sql.peer_delta_acpl
            existing.peer_delta_match = updated_sql.peer_delta_match
            existing.longest_streak = updated_sql.longest_streak
            existing.selectivity_score = updated_sql.selectivity_score
            existing.time_management = updated_sql.time_management
            existing.opening_patterns = updated_sql.opening_patterns
            existing.performance = updated_sql.performance
            existing.phase_quality = updated_sql.phase_quality
            existing.clutch_accuracy = updated_sql.clutch_accuracy
            existing.endgame = updated_sql.endgame
            existing.time_complexity = updated_sql.time_complexity
            existing.benchmark = updated_sql.benchmark
            existing.risk_score = updated_sql.risk_score
            existing.risk_factors = updated_sql.risk_factors
            existing.confidence_level = updated_sql.confidence_level
            existing.first_game_date = updated_sql.first_game_date
            existing.last_game_date = updated_sql.last_game_date
            existing.analyzed_at = updated_sql.analyzed_at
            existing.suspicious_games_ids = updated_sql.suspicious_games_ids

            sql_analysis = existing
        else:
            sql_analysis = domain_to_analysis(analysis)
            self.session.add(sql_analysis)

        self.session.commit()
        self.session.refresh(sql_analysis)

        return analysis_to_domain(sql_analysis)

    def get_high_risk_players(self, risk_threshold: int = 80) -> List[DomainPlayerAnalysis]:
        """Obtiene jugadores con alto riesgo."""
        statement = select(SQLPlayerAnalysis).where(
            SQLPlayerAnalysis.risk_score >= risk_threshold
        ).order_by(SQLPlayerAnalysis.risk_score.desc())

        sql_analyses = self.session.exec(statement).all()
        return [analysis_to_domain(analysis) for analysis in sql_analyses]

    def get_recent_analyses(self, limit: int = 10) -> List[DomainPlayerAnalysis]:
        """Obtiene análisis recientes."""
        statement = select(SQLPlayerAnalysis).order_by(
            SQLPlayerAnalysis.analyzed_at.desc()
        ).limit(limit)

        sql_analyses = self.session.exec(statement).all()
        return [analysis_to_domain(analysis) for analysis in sql_analyses]

    def count_analyses_by_risk_level(self) -> dict:
        """Cuenta análisis por nivel de riesgo."""
        all_analyses = self.session.exec(select(SQLPlayerAnalysis)).all()

        counts = {
            "low_risk": 0,      # 0-30
            "medium_risk": 0,   # 31-70
            "high_risk": 0,     # 71-100
            "total": len(all_analyses)
        }

        for analysis in all_analyses:
            if analysis.risk_score <= 30:
                counts["low_risk"] += 1
            elif analysis.risk_score <= 70:
                counts["medium_risk"] += 1
            else:
                counts["high_risk"] += 1

        return counts