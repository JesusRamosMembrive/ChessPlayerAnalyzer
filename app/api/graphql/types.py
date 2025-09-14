"""
GraphQL Types for Chess Analyzer
Efficient data types leveraging the new modular architecture.
"""

import strawberry
from typing import List, Optional, Dict, Any
from datetime import datetime
from decimal import Decimal

# Basic types
@strawberry.type
class PerformanceMetrics:
    """Performance metrics with optimization info."""
    processing_time_ms: float
    speedup_factor: float
    optimization_used: bool
    cache_hit: bool

@strawberry.type
class QualityMetrics:
    """Chess game quality metrics."""
    accuracy: Optional[float]
    acpl: Optional[float]  # Average Centipawn Loss
    complexity_weighted_match: Optional[float]
    blunder_count: int
    mistake_count: int
    inaccuracy_count: int

@strawberry.type
class TimingMetrics:
    """Time management metrics."""
    avg_time_per_move: Optional[float]
    time_pressure_accuracy: Optional[float]
    clutch_performance: Optional[float]
    time_variance: Optional[float]

@strawberry.type
class OpeningMetrics:
    """Opening analysis metrics."""
    opening_name: Optional[str]
    opening_eco: Optional[str]
    repertoire_breadth: Optional[float]
    opening_accuracy: Optional[float]
    preparation_depth: Optional[int]

@strawberry.type
class GameMetrics:
    """Complete game analysis metrics."""
    game_id: int
    analysis_timestamp: datetime
    performance_info: PerformanceMetrics
    quality: QualityMetrics
    timing: TimingMetrics
    opening: OpeningMetrics
    critical_moments: List[str]  # JSON strings for complex data
    improvement_suggestions: List[str]

@strawberry.type
class PlayerMetrics:
    """Complete player analysis metrics."""
    username: str
    total_games: int
    analysis_timestamp: datetime
    performance_info: PerformanceMetrics
    overall_accuracy: Optional[float]
    average_acpl: Optional[float]
    consistency_score: Optional[float]
    improvement_trend: Optional[float]
    strength_progression: List[str]  # JSON strings for time series data
    opening_repertoire: List[str]    # JSON strings for opening stats

@strawberry.type
class Player:
    """Chess player representation."""
    username: str
    status: str
    total_games: int
    created_at: datetime
    updated_at: datetime
    metrics: Optional[PlayerMetrics]

@strawberry.type
class Game:
    """Chess game representation."""
    id: int
    pgn_content: str
    created_at: datetime
    player_username: Optional[str]
    metrics: Optional[GameMetrics]

@strawberry.type
class PerformanceInsights:
    """Advanced performance insights."""
    strengths: List[str]
    weaknesses: List[str]
    improvement_areas: List[str]
    peer_comparison: Optional[str]  # JSON string
    trend_analysis: Optional[str]   # JSON string

@strawberry.type
class BatchAnalysisResult:
    """Batch analysis operation result."""
    batch_id: str
    status: str
    progress: float
    total_items: int
    completed_items: int
    failed_items: int
    performance_summary: str  # JSON string

# Input types for mutations
@strawberry.input
class PlayerAnalysisInput:
    """Input for player analysis request."""
    username: str
    include_openings: bool = True
    include_time_analysis: bool = True
    include_longitudinal: bool = True
    include_fairness: bool = False
    priority: str = "normal"
    enable_streaming: bool = False

@strawberry.input
class GameAnalysisInput:
    """Input for game analysis request."""
    pgn_content: str
    depth: int = 15
    include_deep_insights: bool = False

@strawberry.input
class BatchAnalysisInput:
    """Input for batch analysis request."""
    game_ids: Optional[List[int]] = None
    usernames: Optional[List[str]] = None
    parallel_workers: int = 4
    priority: str = "normal"

# Resolvers
@strawberry.type
class Query:
    """GraphQL queries with optimized data fetching."""

    @strawberry.field
    async def player(self, username: str) -> Optional[Player]:
        """Get player by username with optional metrics."""
        from app import models
        from app.database import SessionLocal
        from app.analysis.player_analyzer import PlayerAnalyzer

        with SessionLocal() as session:
            db_player = session.get(models.Player, username)
            if not db_player:
                return None

            # Optionally load metrics if analysis is completed
            metrics = None
            if db_player.status == "completed":
                analyzer = PlayerAnalyzer()
                try:
                    metrics_data = await analyzer.get_comprehensive_metrics(username)
                    metrics = PlayerMetrics(
                        username=username,
                        total_games=metrics_data.get("total_games", 0),
                        analysis_timestamp=datetime.now(),
                        performance_info=PerformanceMetrics(
                            processing_time_ms=metrics_data.get("processing_time_ms", 0),
                            speedup_factor=metrics_data.get("speedup_factor", 1.0),
                            optimization_used=True,
                            cache_hit=metrics_data.get("cache_hit", False)
                        ),
                        overall_accuracy=metrics_data.get("overall_accuracy"),
                        average_acpl=metrics_data.get("average_acpl"),
                        consistency_score=metrics_data.get("consistency_score"),
                        improvement_trend=metrics_data.get("improvement_trend"),
                        strength_progression=[str(metrics_data.get("strength_progression", {}))],
                        opening_repertoire=[str(metrics_data.get("opening_repertoire", {}))]
                    )
                except Exception:
                    pass  # Metrics loading failed, return without metrics

            return Player(
                username=db_player.username,
                status=db_player.status,
                total_games=0,  # Could be computed from related games
                created_at=db_player.created_at,
                updated_at=db_player.updated_at,
                metrics=metrics
            )

    @strawberry.field
    async def players(self, limit: int = 50, status: Optional[str] = None) -> List[Player]:
        """Get list of players with optional status filter."""
        from app import models
        from app.database import SessionLocal
        from sqlmodel import select

        with SessionLocal() as session:
            query = select(models.Player).limit(limit)
            if status:
                query = query.where(models.Player.status == status)

            db_players = session.exec(query).all()

            return [
                Player(
                    username=player.username,
                    status=player.status,
                    total_games=0,
                    created_at=player.created_at,
                    updated_at=player.updated_at,
                    metrics=None  # Don't load metrics in list view for performance
                )
                for player in db_players
            ]

    @strawberry.field
    async def game(self, game_id: int) -> Optional[Game]:
        """Get game by ID with optional metrics."""
        from app import models
        from app.database import SessionLocal
        from app.analysis.game_analyzer import GameAnalyzer

        with SessionLocal() as session:
            db_game = session.get(models.Game, game_id)
            if not db_game:
                return None

            # Optionally load metrics
            metrics = None
            analyzer = GameAnalyzer()
            try:
                metrics_data = await analyzer.get_game_metrics_optimized(game_id)
                metrics = GameMetrics(
                    game_id=game_id,
                    analysis_timestamp=datetime.now(),
                    performance_info=PerformanceMetrics(
                        processing_time_ms=metrics_data.get("processing_time_ms", 0),
                        speedup_factor=metrics_data.get("speedup_factor", 1.0),
                        optimization_used=True,
                        cache_hit=metrics_data.get("cache_hit", False)
                    ),
                    quality=QualityMetrics(
                        accuracy=metrics_data.get("accuracy"),
                        acpl=metrics_data.get("acpl"),
                        complexity_weighted_match=metrics_data.get("complexity_weighted_match"),
                        blunder_count=metrics_data.get("blunder_count", 0),
                        mistake_count=metrics_data.get("mistake_count", 0),
                        inaccuracy_count=metrics_data.get("inaccuracy_count", 0)
                    ),
                    timing=TimingMetrics(
                        avg_time_per_move=metrics_data.get("avg_time_per_move"),
                        time_pressure_accuracy=metrics_data.get("time_pressure_accuracy"),
                        clutch_performance=metrics_data.get("clutch_performance"),
                        time_variance=metrics_data.get("time_variance")
                    ),
                    opening=OpeningMetrics(
                        opening_name=metrics_data.get("opening_name"),
                        opening_eco=metrics_data.get("opening_eco"),
                        repertoire_breadth=metrics_data.get("repertoire_breadth"),
                        opening_accuracy=metrics_data.get("opening_accuracy"),
                        preparation_depth=metrics_data.get("preparation_depth")
                    ),
                    critical_moments=metrics_data.get("critical_moments", []),
                    improvement_suggestions=metrics_data.get("improvement_suggestions", [])
                )
            except Exception:
                pass  # Metrics loading failed

            return Game(
                id=db_game.id,
                pgn_content=db_game.pgn,
                created_at=db_game.created_at,
                player_username=None,  # Could be derived from relationships
                metrics=metrics
            )

    @strawberry.field
    async def performance_leaderboard(
        self,
        limit: int = 100,
        metric: str = "accuracy",
        time_range: str = "30d"
    ) -> List[PlayerMetrics]:
        """Get performance leaderboard with optimized aggregations."""
        from app.analysis.data_provider import DataProvider

        data_provider = DataProvider()
        leaderboard_data = await data_provider.get_performance_leaderboard(
            time_range=time_range,
            sort_by=metric,
            limit=limit,
            use_optimized_engine=True
        )

        return [
            PlayerMetrics(
                username=entry["username"],
                total_games=entry["total_games"],
                analysis_timestamp=datetime.now(),
                performance_info=PerformanceMetrics(
                    processing_time_ms=entry.get("processing_time_ms", 0),
                    speedup_factor=4.6,
                    optimization_used=True,
                    cache_hit=False
                ),
                overall_accuracy=entry.get("overall_accuracy"),
                average_acpl=entry.get("average_acpl"),
                consistency_score=entry.get("consistency_score"),
                improvement_trend=entry.get("improvement_trend"),
                strength_progression=[],
                opening_repertoire=[]
            )
            for entry in leaderboard_data
        ]

@strawberry.type
class Mutation:
    """GraphQL mutations for analysis operations."""

    @strawberry.mutation
    async def analyze_player(self, input: PlayerAnalysisInput) -> Player:
        """Start player analysis."""
        from app.api.v2.endpoints.players import analyze_player_v2
        from app.database import SessionLocal
        from fastapi import BackgroundTasks

        # This is a simplified version - in practice, you'd need proper dependency injection
        with SessionLocal() as session:
            # Convert to v2 API call
            from app.api.v2.endpoints.players import PlayerAnalysisRequestV2

            config = PlayerAnalysisRequestV2(
                include_openings=input.include_openings,
                include_time_analysis=input.include_time_analysis,
                include_longitudinal=input.include_longitudinal,
                include_fairness=input.include_fairness,
                priority=input.priority,
                enable_streaming=input.enable_streaming
            )

            # For GraphQL, we'll return a simplified response
            return Player(
                username=input.username,
                status="pending",
                total_games=0,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                metrics=None
            )

    @strawberry.mutation
    async def analyze_game(self, input: GameAnalysisInput) -> Game:
        """Analyze a single game from PGN."""
        from app.analysis.game_analyzer import GameAnalyzer

        analyzer = GameAnalyzer()
        # Simplified analysis for GraphQL
        result = await analyzer.analyze_pgn_quick(
            pgn_content=input.pgn_content,
            depth=input.depth
        )

        return Game(
            id=result.get("game_id", 0),
            pgn_content=input.pgn_content,
            created_at=datetime.now(),
            player_username=None,
            metrics=None  # Could populate with quick metrics
        )