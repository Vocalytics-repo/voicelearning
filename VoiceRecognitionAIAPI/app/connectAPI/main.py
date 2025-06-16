from fastapi import FastAPI
import uvicorn
import os
from dotenv import load_dotenv
load_dotenv()  # .env 파일 로드
# 라우터 임포트
from app.connectAPI.routers.stt import router as stt_router
# 전역 범위에서 FastAPI 앱 생성
app = FastAPI(title="WAV to Text API")

'''# 서버 시작 함수
def start_server():
    # 라우터 등록
    app.include_router(stt_router)
    return app

# 메인 실행 블록
if __name__ == "__main__":
    app = start_server()  # 앱 초기화
    uvicorn.run(app, host="0.0.0.0", port=8081)'''
# 라우터를 바로 등록 (조건부가 아니라 항상)
app.include_router(stt_router)

# 메인 실행 블록
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)