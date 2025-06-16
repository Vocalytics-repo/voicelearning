# Python 이미지 사용
FROM python:3.9

# 시스템 패키지 설치 (pydub용 ffmpeg 추가)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# gdown 설치
RUN pip install gdown

# 모델 파일 저장할 디렉토리 생성
RUN mkdir -p /app/app/connectAPI/service/model/

# Hugging Face 모델 미리 다운로드
RUN python -c "from transformers import pipeline; pipeline('audio-classification', model='alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech')"

# 훈련된 모델 복사
COPY app/connectAPI/service/model/trainedModel.pt /app/app/connectAPI/service/model/trainedModel.pt

# 애플리케이션 코드 복사
COPY . .

ENV PYTHONPATH=/app

# 컨테이너 실행 시 실행할 명령어
CMD ["python", "app/connectAPI/main.py"]