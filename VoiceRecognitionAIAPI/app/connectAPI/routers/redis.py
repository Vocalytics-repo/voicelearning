from fastapi import APIRouter, HTTPException

router = APIRouter()

redis_client = None

@router.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """
    변환 작업 상태 확인 API
    """
    redis_key = f"pcm:{job_id}"
    if not redis_client.exists(redis_key):
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")

    # 상태 확인 로직 구현
    return {"job_id": job_id, "status": "processing"}