from fastapi import FastAPI
import uvicorn
import os

# 라우터 임포트
from app.connectAPI.routers.wav_to_pcm import router as wav_to_pcm_router
from app.connectAPI.routers.pcm_to_text import router as pcm_to_text_router
# 전역 범위에서 FastAPI 앱 생성
app = FastAPI(title="WAV to Text API")

# 임시 디렉토리 설정
TEMP_DIRECTORY = "temp"


# 서버 시작 함수
def start_server():
    # 임시 디렉토리 생성 (없는 경우)
    os.makedirs(TEMP_DIRECTORY, exist_ok=True)

    # wav_to_pcm.py 모듈의 전역 변수 설정
    import app.connectAPI.routers.wav_to_pcm as wav_to_pcm_module
    wav_to_pcm_module.TEMP_DIR = TEMP_DIRECTORY

    # 라우터 등록
    app.include_router(wav_to_pcm_router)
    app.include_router(pcm_to_text_router)

    return app


# 메인 실행 블록
if __name__ == "__main__":
    app = start_server()  # 앱 초기화
    uvicorn.run(app, host="0.0.0.0", port=8000)