# voicelearning STT API

## ⚙️ Tech Stack

- **Framework**: FastAPI
- **Speech Recognition**: Whisper (base model / Fine tunning), Pytorch, HuggingFace
- **Audio Handling**: NumPy, wave
- **Containerization**: Docker, docker-compose

## Project Structure
```
VoiceRecognitionAIAPI/
├── .venv/
├── app/
│   ├── connectAPI/
│   │   ├── routers/
│   │   │   ├── __init__.py
│   │   │   ├── struct.py
│   │   │   └── stt.py
│   │   ├── service/
│   │   │   ├── errorcheck/
│   │   │   │   └── typeerror.py
│   │   │   ├── model/
│   │   │   │   ├── __init__.py
│   │   │   │   └── models.py
│   │   │   ├── preprocessing/
│   │   │   │   └── __init__.py
│   │   │   ├── __init__.py
│   │   │   └── control.py
│   │   └── temp/
│   │       ├── __init__.py
│   │       └── main.py
│   └── __init__.py
├── Managejupyter/
│   ├── __init__.py
│   └── temp/
├── venv/
│   ├── bin/
│   ├── include/
│   ├── lib/
│   ├── share/
│   └── pyvenv.cfg
├── .gitignore
├── docker-compose.yml
└── trainedModel.pt
```
