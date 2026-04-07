from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.src.api.schemas.base import BaseModelFlex
from backend.src.tasks.broker import broker

router = APIRouter(prefix="/ingestion", tags=["Ingestion"])


class JobStatusResponse(BaseModelFlex):
    job_id: str
    status: str
    progress: float
    result: dict[str, object] | None = None
    error: str | None = None


@router.get("/status/{job_id}")
async def get_job_status(job_id: str) -> JobStatusResponse:
    task = broker.get_task(job_id)

    if task is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if task.ready():
        if task.is_failed():
            error = task.get_error()
            return JobStatusResponse(
                job_id=job_id,
                status="failed",
                progress=1.0,
                result=None,
                error=str(error) if error else "Unknown error",
            )
        else:
            result = task.get_result()
            return JobStatusResponse(
                job_id=job_id,
                status="completed",
                progress=1.0,
                result=result if isinstance(result, dict) else {"result": result},
                error=None,
            )

    return JobStatusResponse(
        job_id=job_id,
        status="running",
        progress=0.0,
        result=None,
        error=None,
    )
