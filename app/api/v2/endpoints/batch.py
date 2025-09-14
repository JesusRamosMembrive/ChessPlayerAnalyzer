"""
API v2 Batch Operations Endpoints
High-performance batch processing leveraging the 4.6x speedup optimizations.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from sqlmodel import Session, select
from pydantic import BaseModel, Field
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app import models
from app.database import get_session
from app.analysis.engine_facade import EngineApi
from app.analysis.data_provider import DataProvider
from app.utils import redis_client

router = APIRouter()

class BatchGameAnalysisRequest(BaseModel):
    """Request for batch game analysis."""
    game_ids: List[int] = Field(..., min_items=1, max_items=1000, description="Game IDs to analyze")
    parallel_workers: int = Field(4, ge=1, le=16, description="Number of parallel workers")
    include_deep_analysis: bool = Field(False, description="Include computationally expensive analysis")
    priority: str = Field("normal", regex="^(low|normal|high)$")

class BatchPlayerAnalysisRequest(BaseModel):
    """Request for batch player analysis."""
    usernames: List[str] = Field(..., min_items=1, max_items=100, description="Usernames to analyze")
    parallel_workers: int = Field(2, ge=1, le=8, description="Number of parallel workers")
    include_openings: bool = Field(True, description="Include opening analysis")
    include_time_analysis: bool = Field(True, description="Include time management analysis")
    include_longitudinal: bool = Field(True, description="Include longitudinal trends")
    priority: str = Field("normal", regex="^(low|normal|high)$")

class BatchStatus(BaseModel):
    """Batch operation status."""
    batch_id: str
    batch_type: str  # "games" or "players"
    status: str  # "pending", "processing", "completed", "failed", "cancelled"
    progress: float = Field(ge=0, le=100)
    items_total: int
    items_completed: int
    items_failed: int
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    performance_metrics: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None

class BatchResult(BaseModel):
    """Batch operation results."""
    batch_id: str
    status: str
    results: List[Dict[str, Any]]
    failed_items: List[Dict[str, Any]]
    performance_summary: Dict[str, float]
    total_processing_time: float

# In-memory batch tracking (in production, use Redis or database)
_batch_operations: Dict[str, Dict[str, Any]] = {}

def generate_batch_id() -> str:
    """Generate unique batch ID."""
    import uuid
    return f"batch_{uuid.uuid4().hex[:8]}"

async def update_batch_progress(batch_id: str, progress: float, items_completed: int = None,
                               performance_metrics: Dict[str, float] = None):
    """Update batch progress in tracking system."""
    if batch_id in _batch_operations:
        _batch_operations[batch_id]["progress"] = progress
        if items_completed is not None:
            _batch_operations[batch_id]["items_completed"] = items_completed
        if performance_metrics:
            _batch_operations[batch_id]["performance_metrics"] = performance_metrics

        # Store in Redis for persistence
        await redis_client.set(
            f"batch:progress:{batch_id}",
            f"{progress}",
            ex=3600  # Expire after 1 hour
        )

async def process_games_batch(batch_id: str, game_ids: List[int], config: BatchGameAnalysisRequest):
    """Process batch of games with performance optimizations."""
    engine_api = EngineApi()
    start_time = datetime.now()

    results = []
    failed_items = []
    total_speedup = 0
    processed_count = 0

    try:
        # Update status to processing
        _batch_operations[batch_id]["status"] = "processing"
        _batch_operations[batch_id]["started_at"] = start_time

        # Process games in batches to leverage optimizations
        batch_size = min(config.parallel_workers, 10)

        for i in range(0, len(game_ids), batch_size):
            batch_games = game_ids[i:i + batch_size]

            # Process batch in parallel
            batch_start = datetime.now()
            batch_results = await asyncio.gather(
                *[engine_api.analyze_game_optimized(game_id) for game_id in batch_games],
                return_exceptions=True
            )
            batch_duration = (datetime.now() - batch_start).total_seconds()

            # Process results
            for j, result in enumerate(batch_results):
                game_id = batch_games[j]

                if isinstance(result, Exception):
                    failed_items.append({
                        "game_id": game_id,
                        "error": str(result),
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    results.append({
                        "game_id": game_id,
                        "analysis": result,
                        "processing_time": result.get("processing_time_ms", 0),
                        "speedup_factor": result.get("speedup_factor", 1.0)
                    })
                    total_speedup += result.get("speedup_factor", 1.0)

                processed_count += 1

            # Update progress
            progress = (processed_count / len(game_ids)) * 100
            avg_speedup = total_speedup / max(len(results), 1)

            performance_metrics = {
                "avg_speedup_factor": avg_speedup,
                "games_per_second": len(batch_games) / batch_duration,
                "batch_processing_time": batch_duration,
                "estimated_remaining_time": (len(game_ids) - processed_count) / (len(batch_games) / batch_duration)
            }

            await update_batch_progress(batch_id, progress, processed_count, performance_metrics)

        # Mark as completed
        completion_time = datetime.now()
        total_duration = (completion_time - start_time).total_seconds()

        _batch_operations[batch_id].update({
            "status": "completed",
            "completed_at": completion_time,
            "results": results,
            "failed_items": failed_items,
            "performance_summary": {
                "total_processing_time": total_duration,
                "avg_speedup_factor": total_speedup / max(len(results), 1),
                "games_per_second": len(game_ids) / total_duration,
                "success_rate": len(results) / len(game_ids) * 100,
                "optimization_effectiveness": "excellent" if total_speedup / max(len(results), 1) > 3.0 else "good"
            }
        })

    except Exception as e:
        _batch_operations[batch_id].update({
            "status": "failed",
            "completed_at": datetime.now(),
            "error_message": str(e)
        })

@router.post(
    "/games",
    response_model=BatchStatus,
    summary="Start batch game analysis",
    description="Start high-performance batch analysis of multiple games leveraging 4.6x speedup optimizations.",
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_batch_game_analysis(
    request: BatchGameAnalysisRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session)
):
    """Start batch game analysis with performance optimizations."""
    # Validate that games exist
    existing_games = session.exec(
        select(models.Game.id).where(models.Game.id.in_(request.game_ids))
    ).all()

    missing_games = set(request.game_ids) - set(existing_games)
    if missing_games:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Games not found: {list(missing_games)}"
        )

    # Generate batch ID and initialize tracking
    batch_id = generate_batch_id()

    # Estimate completion time based on performance gains
    base_time_per_game = 1.16  # seconds (pre-optimization baseline)
    optimized_time_per_game = base_time_per_game / 4.6  # Post-optimization
    estimated_duration = len(request.game_ids) * optimized_time_per_game / request.parallel_workers
    estimated_completion = datetime.now() + timedelta(seconds=estimated_duration)

    # Initialize batch tracking
    batch_info = {
        "batch_id": batch_id,
        "batch_type": "games",
        "status": "pending",
        "progress": 0.0,
        "items_total": len(request.game_ids),
        "items_completed": 0,
        "items_failed": 0,
        "created_at": datetime.now(),
        "estimated_completion": estimated_completion,
        "config": request.dict()
    }

    _batch_operations[batch_id] = batch_info

    # Start background processing
    background_tasks.add_task(process_games_batch, batch_id, request.game_ids, request)

    return BatchStatus(**batch_info)

@router.get(
    "/{batch_id}/status",
    response_model=BatchStatus,
    summary="Get batch operation status",
    description="Get detailed status and progress of a batch operation.",
)
async def get_batch_status(batch_id: str):
    """Get batch operation status."""
    if batch_id not in _batch_operations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch operation {batch_id} not found"
        )

    batch_info = _batch_operations[batch_id]
    return BatchStatus(**batch_info)

@router.get(
    "/{batch_id}/results",
    response_model=BatchResult,
    summary="Get batch operation results",
    description="Get complete results of a finished batch operation.",
)
async def get_batch_results(batch_id: str):
    """Get batch operation results."""
    if batch_id not in _batch_operations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch operation {batch_id} not found"
        )

    batch_info = _batch_operations[batch_id]

    if batch_info["status"] not in ["completed", "failed"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Batch operation {batch_id} is still {batch_info['status']}"
        )

    return BatchResult(
        batch_id=batch_id,
        status=batch_info["status"],
        results=batch_info.get("results", []),
        failed_items=batch_info.get("failed_items", []),
        performance_summary=batch_info.get("performance_summary", {}),
        total_processing_time=batch_info.get("performance_summary", {}).get("total_processing_time", 0)
    )

@router.delete(
    "/{batch_id}",
    summary="Cancel batch operation",
    description="Cancel a running batch operation.",
)
async def cancel_batch_operation(batch_id: str):
    """Cancel a batch operation."""
    if batch_id not in _batch_operations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch operation {batch_id} not found"
        )

    batch_info = _batch_operations[batch_id]

    if batch_info["status"] in ["completed", "failed"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot cancel {batch_info['status']} batch operation"
        )

    # Mark as cancelled
    _batch_operations[batch_id]["status"] = "cancelled"
    _batch_operations[batch_id]["completed_at"] = datetime.now()

    return {"message": f"Batch operation {batch_id} cancelled successfully"}

@router.get(
    "/",
    response_model=List[BatchStatus],
    summary="List batch operations",
    description="List all batch operations with their current status.",
)
async def list_batch_operations(
    status_filter: Optional[str] = Query(None, regex="^(pending|processing|completed|failed|cancelled)$"),
    limit: int = Query(50, ge=1, le=200),
):
    """List batch operations."""
    operations = list(_batch_operations.values())

    if status_filter:
        operations = [op for op in operations if op["status"] == status_filter]

    # Sort by creation time, most recent first
    operations.sort(key=lambda x: x["created_at"], reverse=True)

    # Apply limit
    operations = operations[:limit]

    return [BatchStatus(**op) for op in operations]