from fastapi import APIRouter, HTTPException
import numpy as np

router = APIRouter()

# 전역 변수로 선언 (main.py에서 초기화됨)
redis_client = None

# PCM 데이터를 텍스트로 변환하는 함수
async def convert_pcm_to_text(pcm_array):
    # 실제 구현에서는 여기에 모델 호출 등의 로직이 들어갑니다
    # 예시 코드입니다
    return "example data"

@router.get("/api/pcm-to-text/{job_id}")
async def process_pcm_to_text(job_id: str) -> str:
    """
    Redis에서 PCM 데이터를 가져와 텍스트로 변환
    """
    redis_key = f"pcm:{job_id}"
    pcm_bytes = redis_client.get(redis_key)

    if not pcm_bytes:
        raise HTTPException(status_code=404, detail="PCM 데이터를 찾을 수 없습니다.")

    # PCM 바이트를 NumPy 배열로 변환 (16비트 PCM 가정)
    pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)

    # 모델을 사용하여 텍스트 변환
    try:
        # 실제 구현에서는 이 부분에 모델 호출 코드가 들어감
        text = await convert_pcm_to_text(pcm_array)
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"텍스트 변환 오류: {str(e)}")