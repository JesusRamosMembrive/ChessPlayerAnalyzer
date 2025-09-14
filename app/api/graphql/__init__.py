"""
GraphQL API for Chess Analyzer
Efficient querying interface leveraging the new modular architecture.
"""

import strawberry
from typing import List, Optional, Dict, Any
from datetime import datetime

from .types import (
    Player,
    Game,
    PlayerMetrics,
    GameMetrics,
    PerformanceInsights,
    Query,
    Mutation
)

# Create the GraphQL schema
schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    description="Chess Analyzer GraphQL API - Efficient querying with 4.6x performance optimizations"
)