"""
API v2 Streaming Endpoints
Real-time streaming analysis updates leveraging WebSocket connections.
"""

from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect, Query
from sqlmodel import Session
import asyncio
import json
import logging

from app import models
from app.database import get_session
from app.utils import redis_client
from app.analysis.engine_facade import EngineApi

router = APIRouter()
logger = logging.getLogger(__name__)

# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for real-time streaming."""

    def __init__(self):
        # Active connections: {client_id: websocket}
        self.active_connections: Dict[str, WebSocket] = {}
        # Subscriptions: {resource_type: {resource_id: [client_ids]}}
        self.subscriptions: Dict[str, Dict[str, List[str]]] = {
            "players": {},
            "games": {},
            "batch": {}
        }

    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        """Remove client connection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]

        # Remove from all subscriptions
        for resource_type in self.subscriptions.values():
            for resource_id in resource_type:
                if client_id in resource_type[resource_id]:
                    resource_type[resource_id].remove(client_id)

        logger.info(f"Client {client_id} disconnected")

    async def subscribe(self, client_id: str, resource_type: str, resource_id: str):
        """Subscribe client to resource updates."""
        if resource_type not in self.subscriptions:
            self.subscriptions[resource_type] = {}

        if resource_id not in self.subscriptions[resource_type]:
            self.subscriptions[resource_type][resource_id] = []

        if client_id not in self.subscriptions[resource_type][resource_id]:
            self.subscriptions[resource_type][resource_id].append(client_id)

        logger.info(f"Client {client_id} subscribed to {resource_type}:{resource_id}")

    async def unsubscribe(self, client_id: str, resource_type: str, resource_id: str):
        """Unsubscribe client from resource updates."""
        if (resource_type in self.subscriptions and
            resource_id in self.subscriptions[resource_type] and
            client_id in self.subscriptions[resource_type][resource_id]):

            self.subscriptions[resource_type][resource_id].remove(client_id)
            logger.info(f"Client {client_id} unsubscribed from {resource_type}:{resource_id}")

    async def broadcast_to_resource(self, resource_type: str, resource_id: str, message: Dict[str, Any]):
        """Send message to all clients subscribed to a resource."""
        if (resource_type in self.subscriptions and
            resource_id in self.subscriptions[resource_type]):

            subscribers = self.subscriptions[resource_type][resource_id]
            disconnected_clients = []

            for client_id in subscribers:
                if client_id in self.active_connections:
                    try:
                        await self.active_connections[client_id].send_text(json.dumps(message))
                    except Exception as e:
                        logger.error(f"Failed to send to client {client_id}: {e}")
                        disconnected_clients.append(client_id)
                else:
                    disconnected_clients.append(client_id)

            # Clean up disconnected clients
            for client_id in disconnected_clients:
                self.subscriptions[resource_type][resource_id].remove(client_id)

    async def send_personal_message(self, client_id: str, message: Dict[str, Any]):
        """Send message to specific client."""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send personal message to {client_id}: {e}")

# Global connection manager
manager = ConnectionManager()

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time streaming."""
    await manager.connect(websocket, client_id)

    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            message = json.loads(data)

            message_type = message.get("type")
            payload = message.get("payload", {})

            if message_type == "subscribe":
                resource_type = payload.get("resource_type")
                resource_id = payload.get("resource_id")

                if resource_type and resource_id:
                    await manager.subscribe(client_id, resource_type, resource_id)
                    await manager.send_personal_message(client_id, {
                        "type": "subscription_confirmed",
                        "payload": {"resource_type": resource_type, "resource_id": resource_id}
                    })

            elif message_type == "unsubscribe":
                resource_type = payload.get("resource_type")
                resource_id = payload.get("resource_id")

                if resource_type and resource_id:
                    await manager.unsubscribe(client_id, resource_type, resource_id)
                    await manager.send_personal_message(client_id, {
                        "type": "unsubscription_confirmed",
                        "payload": {"resource_type": resource_type, "resource_id": resource_id}
                    })

            elif message_type == "ping":
                await manager.send_personal_message(client_id, {
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                })

    except WebSocketDisconnect:
        manager.disconnect(client_id)

@router.get(
    "/players/{username}",
    summary="Stream player analysis progress",
    description="Server-Sent Events endpoint for real-time player analysis updates.",
)
async def stream_player_analysis(
    username: str,
    session: Session = Depends(get_session)
):
    """Stream player analysis progress using Server-Sent Events."""
    player = session.get(models.Player, username)
    if not player:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Player {username} not found"
        )

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate Server-Sent Events for player analysis progress."""
        last_progress = -1
        last_update_time = datetime.min

        while True:
            try:
                # Get current progress from Redis
                progress_key = f"player:progress:{username}"
                progress_data = await redis_client.get(progress_key)

                if progress_data:
                    progress_info = json.loads(progress_data)
                    current_progress = progress_info.get("progress_percentage", 0)
                    current_phase = progress_info.get("current_phase", "unknown")
                    last_update = datetime.fromisoformat(progress_info.get("last_update", datetime.now().isoformat()))

                    # Send update if progress changed or significant time passed
                    if (current_progress != last_progress or
                        (datetime.now() - last_update_time).seconds > 30):

                        # Calculate performance metrics
                        games_analyzed = progress_info.get("games_analyzed", 0)
                        games_total = progress_info.get("games_total", 0)
                        analysis_rate = progress_info.get("games_per_minute", 0)
                        speedup_factor = progress_info.get("speedup_factor", 1.0)

                        # Estimate remaining time
                        remaining_games = games_total - games_analyzed
                        estimated_remaining_minutes = remaining_games / analysis_rate if analysis_rate > 0 else 0

                        event_data = {
                            "type": "progress_update",
                            "timestamp": datetime.now().isoformat(),
                            "data": {
                                "username": username,
                                "progress_percentage": current_progress,
                                "current_phase": current_phase,
                                "games_analyzed": games_analyzed,
                                "games_total": games_total,
                                "analysis_rate_per_minute": analysis_rate,
                                "speedup_factor": speedup_factor,
                                "estimated_remaining_minutes": estimated_remaining_minutes,
                                "performance_grade": (
                                    "excellent" if speedup_factor >= 4.0 else
                                    "good" if speedup_factor >= 2.5 else
                                    "normal"
                                )
                            }
                        }

                        # Format as Server-Sent Event
                        yield f"data: {json.dumps(event_data)}\n\n"

                        last_progress = current_progress
                        last_update_time = datetime.now()

                    # Check if analysis completed
                    if current_progress >= 100 or progress_info.get("status") == "completed":
                        completion_event = {
                            "type": "analysis_completed",
                            "timestamp": datetime.now().isoformat(),
                            "data": {
                                "username": username,
                                "final_progress": 100,
                                "total_games_analyzed": games_total,
                                "final_speedup_factor": speedup_factor,
                                "results_url": f"/api/v2/players/{username}/insights"
                            }
                        }
                        yield f"data: {json.dumps(completion_event)}\n\n"
                        break

                else:
                    # No progress data available
                    if last_progress != -1:  # Only send once
                        no_data_event = {
                            "type": "no_data",
                            "timestamp": datetime.now().isoformat(),
                            "data": {
                                "username": username,
                                "message": "No analysis in progress"
                            }
                        }
                        yield f"data: {json.dumps(no_data_event)}\n\n"
                        last_progress = -1

                # Wait before next check
                await asyncio.sleep(5)

            except Exception as e:
                error_event = {
                    "type": "error",
                    "timestamp": datetime.now().isoformat(),
                    "data": {
                        "username": username,
                        "error": str(e)
                    }
                }
                yield f"data: {json.dumps(error_event)}\n\n"
                break

    # Return the event generator (FastAPI will handle SSE formatting)
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

@router.post(
    "/notify/player/{username}",
    summary="Send notification to player analysis subscribers",
    description="Send custom notification to all clients streaming a player's analysis.",
)
async def notify_player_subscribers(
    username: str,
    message: Dict[str, Any],
    notification_type: str = Query("custom", description="Type of notification")
):
    """Send notification to all subscribers of a player's analysis."""
    notification = {
        "type": notification_type,
        "timestamp": datetime.now().isoformat(),
        "data": {
            "username": username,
            **message
        }
    }

    await manager.broadcast_to_resource("players", username, notification)

    return {
        "message": f"Notification sent to all subscribers of player {username}",
        "notification_type": notification_type,
        "timestamp": datetime.now().isoformat()
    }

@router.get(
    "/performance/live",
    summary="Live performance metrics stream",
    description="Stream live performance metrics showing optimization effectiveness.",
)
async def stream_performance_metrics():
    """Stream live performance metrics."""
    async def performance_generator() -> AsyncGenerator[str, None]:
        """Generate performance metrics events."""
        while True:
            try:
                # Get current performance metrics from Redis or monitoring system
                # This is a placeholder - in production, pull from actual monitoring
                performance_data = {
                    "type": "performance_metrics",
                    "timestamp": datetime.now().isoformat(),
                    "data": {
                        "current_analysis_rate": 8.5,  # games per minute
                        "avg_speedup_factor": 4.2,
                        "active_optimizations": {
                            "numpy_quality": {"active": True, "speedup": 6.7},
                            "numpy_timing": {"active": True, "speedup": 2.4},
                            "numpy_longitudinal": {"active": True, "speedup": 2.4}
                        },
                        "system_health": {
                            "cpu_usage": 45.2,
                            "memory_usage": 67.8,
                            "queue_length": 12
                        },
                        "performance_grade": "excellent"
                    }
                }

                yield f"data: {json.dumps(performance_data)}\n\n"

                # Update every 10 seconds
                await asyncio.sleep(10)

            except Exception as e:
                error_event = {
                    "type": "error",
                    "timestamp": datetime.now().isoformat(),
                    "data": {"error": str(e)}
                }
                yield f"data: {json.dumps(error_event)}\n\n"
                break

    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        performance_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*"
        }
    )