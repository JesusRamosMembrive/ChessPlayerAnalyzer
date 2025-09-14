"""
API v2 - Enhanced Chess Analyzer API
Leverages the new modular architecture (engine_facade, data_provider, etc.)
with improved performance and advanced features while maintaining backward compatibility.
"""

from fastapi import APIRouter

# Import all v2 endpoint routers
from .endpoints import (
    players,
    games,
    analysis,
    health,
    tasks,
    batch,      # New: Batch operations
    streaming,  # New: Streaming analysis
    aggregates, # New: Advanced aggregations
    graphql_endpoint, # New: GraphQL endpoint
)

# Create the API v2 router
api_router = APIRouter(
    prefix="/v2",
    tags=["v2"],
    responses={
        500: {"description": "Internal server error"},
        429: {"description": "Rate limit exceeded"},
    }
)

# Include all endpoint routers with proper prefixes
api_router.include_router(health.router, prefix="/health", tags=["health", "v2"])
api_router.include_router(players.router, prefix="/players", tags=["players", "v2"])
api_router.include_router(games.router, prefix="/games", tags=["games", "v2"])
api_router.include_router(analysis.router, prefix="/analysis", tags=["analysis", "v2"])
api_router.include_router(tasks.router, prefix="/tasks", tags=["tasks", "v2"])

# New v2-specific endpoints
api_router.include_router(batch.router, prefix="/batch", tags=["batch", "v2"])
api_router.include_router(streaming.router, prefix="/streaming", tags=["streaming", "v2"])
api_router.include_router(aggregates.router, prefix="/aggregates", tags=["aggregates", "v2"])
api_router.include_router(graphql_endpoint.router, prefix="/graphql", tags=["graphql", "v2"])