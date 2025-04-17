# voicelearning STT API

## ⚙️ Tech Stack

- **Framework**: FastAPI
- **Speech Recognition**: Whisper (base model / Fine tunning), Pytorch, HuggingFace
- **Audio Handling**: NumPy, wave
- **Containerization**: Docker, docker-compose

## Project Structure
VoiceRecognitionAIAPI/
├── .venv/
├── app/
│   ├── connectAPI/
│   │   └── manage/
│   │       └── manage_redis.py
│   ├── model/
│   │   └── models.py
│   ├── routers/
│   │   ├── pcm_to_text.py
│   │   ├── redis.py
│   │   ├── struct.py
│   │   └── wav_to_pcm.py
│   └── main.py
├── model/
│   └── Dockerfile
├── redis/
│   ├── init.py
│   └── Dockerfile
├── docker-compose.yml
└── Dockerfile
