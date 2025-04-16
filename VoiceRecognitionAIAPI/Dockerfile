# Python 이미지 사용
FROM python:3.9

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

ENV PYTHONPATH=/app

# 컨테이너 실행 시 실행할 명령어
CMD ["python", "app/connectAPI/main.py"]