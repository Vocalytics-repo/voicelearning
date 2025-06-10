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

## API Documentation

### Endpoints

#### POST /api/v1/stt
Speech-to-Text conversion endpoint.

**Request:**
- Content-Type: multipart/form-data
- Body: audio file (.wav, .mp3)

**Response:**
- Status: 200 OK
- Content-Type: application/json
```json
{
  "text": "converted speech text",
  "confidence": 0.95
}


# DataSet
- AIhub korea sound dataset
- URI :  https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=123
- Explain : 
    Korean Free Speech Voice Data for Acoustic Models to Improve Conversational Speech Recognition Performance

    1000 hours of Korean conversation voices from 2,000 people in a quiet environment

    Recording two people talking freely on various topics

- File sturcture:
    - KsponSpeech_01.zip - 14.25GB (Use)
    - KsponSpeech_02.zip - 14.26GB
    - KsponSpeech_03.zip - 14.18GB
    - KsponSpeech_04.zip - 14.23GB
    - KsponSpeech_05.zip - 14.57GB


# Model
- Whisper-small : Whisper is a pre-trained model for automatic speech recognition and speech translation 
Learned based on 680K hours of labeled data
Seq2seq model with transformer-based encoder-decoder model
Learned based on weak supervised learning

For whisper small model

Parameters: 244M (240 million)

Size: 461MB

Language: Supports 99 languages

Reason for selection: Because I wanted to extract performance in a light state rather than a heavy state with multiple services combined in the STT service
