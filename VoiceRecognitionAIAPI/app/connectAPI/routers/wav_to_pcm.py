from fastapi import APIRouter, File, UploadFile, HTTPException
from app.connectAPI.routers.struct import TextResponse
import time
import os
import uuid
import aiofiles
import numpy as np
import wave

router = APIRouter()

TEMP_DIR = None

@router.post("/api/wav-to-text", response_model=TextResponse)
async def wav_to_text(file: UploadFile = File(...)):
    """
    WAV 파일을 받아서 텍스트로 변환하는 API
    """
    start_time = time.time()

    # 1. WAV 파일 받기
    if not file.filename.endswith('.wav'):
        raise HTTPException(status_code=400, detail="WAV 파일만 지원합니다.")

    # 고유 작업 ID 생성
    job_id = str(uuid.uuid4())

    # WAV 파일 임시 저장
    wav_path = os.path.join(TEMP_DIR, f"{job_id}.wav")
    async with aiofiles.open(wav_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)

    # 2. WAV -> PCM 변환
    pcm_data = await convert_wav_to_pcm(wav_path)

    text = "example data"

    # 5. 임시 파일 삭제
    os.remove(wav_path)

    # 6. 응답 반환
    processing_time = time.time() - start_time
    return TextResponse(
        text=text,
        job_id=job_id,
        processing_time=processing_time
    )


async def convert_wav_to_pcm(wav_path: str) -> np.ndarray:
    """
    WAV 파일을 PCM 데이터로 변환
    """
    try:
        with wave.open(wav_path, 'rb') as wav_file:
            # WAV 파일 정보 읽기
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            n_frames = wav_file.getnframes()

            # PCM 데이터 읽기
            pcm_data = wav_file.readframes(n_frames)

            # NumPy 배열로 변환
            if sample_width == 2:  # 16-bit
                pcm_array = np.frombuffer(pcm_data, dtype=np.int16)
            elif sample_width == 4:  # 32-bit
                pcm_array = np.frombuffer(pcm_data, dtype=np.int32)
            else:
                pcm_array = np.frombuffer(pcm_data, dtype=np.uint8)

            return pcm_array
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"WAV -> PCM 변환 오류: {str(e)}")



