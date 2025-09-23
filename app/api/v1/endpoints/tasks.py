from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session
from celery.result import AsyncResult

from app.celery_tasks import celery_app
from app.database import get_session
from app.schemas import TaskStatusOut, TaskCancelOut, TaskResultOut

router = APIRouter()

@router.get(
    "/{task_id}",
    response_model=TaskStatusOut,
    summary="Obtener estado de una tarea Celery",
    description="Devuelve el estado actual, mensaje y progreso de la tarea identificada por **task_id**.",
)
async def get_task_status(task_id: str):
    """Get the status of a Celery task."""
    try:
        res = AsyncResult(task_id, app=celery_app)
        
        if res.state == "PENDING":
            return {
                "task_id": task_id,
                "state": res.state,
                "status": "Task is pending execution"
            }
        elif res.state == "FAILURE":
            return {
                "task_id": task_id,
                "state": res.state,
                "status": "Task failed",
                "error": str(res.result) if res.result else "Unknown error"
            }
        elif res.state == "SUCCESS":
            return {
                "task_id": task_id,
                "state": res.state,
                "status": "Task completed successfully",
                "result": res.result
            }
        else:
            # Task is in progress
            return {
                "task_id": task_id,
                "state": res.state,
                "status": "Task is in progress",
                "progress": getattr(res, "info", {}).get("progress", 0) if hasattr(res, "info") else 0
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting task status: {str(e)}")

@router.delete(
    "/{task_id}",
    response_model=TaskCancelOut,
    summary="Cancelar tarea Celery",
    description="Revoca una tarea en ejecución o pendiente.",
)
async def cancel_task(task_id: str):
    """Cancel a running Celery task."""
    try:
        result = AsyncResult(task_id, app=celery_app)
        
        if result.state in ["PENDING", "STARTED"]:
            # Revoke the task (terminate=True to force termination)
            result.revoke(terminate=True)
            return {
                "task_id": task_id,
                "status": "cancelled",
                "message": "Task has been cancelled successfully"
            }
        else:
            # Task is not running or already completed
            return {
                "task_id": task_id,
                "status": result.state,
                "message": f"Task cannot be cancelled because it's in state: {result.state}"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cancelling task: {str(e)}")

@router.get(
    "/{task_id}/result",
    response_model=TaskResultOut,
    summary="Obtener resultado de tarea Celery",
    description="Devuelve el resultado si la tarea ha finalizado o 202 si sigue en progreso.",
    responses={202: {"description": "Tarea no completada aún"}},
)
async def get_task_result(task_id: str):
    """Get the result of a completed Celery task."""
    try:
        result = AsyncResult(task_id, app=celery_app)
        
        if not result.ready():
            raise HTTPException(
                status_code=202,
                detail={
                    "task_id": task_id,
                    "status": "Task not yet completed",
                    "state": result.state
                }
            )
        
        if result.failed():
            return {
                "task_id": task_id,
                "status": "error",
                "state": result.state,
                "error": str(result.result)
            }
        
        return {
            "task_id": task_id,
            "status": "success",
            "state": result.state,
            "result": result.result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting task result: {str(e)}")
