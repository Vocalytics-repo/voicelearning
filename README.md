# VoiceRecognitionAIAPI

A robust Speech-to-Text (STT) API service built with FastAPI and Whisper, designed for high-performance Korean speech recognition with fine-tuned models.

## ⚙️ Tech Stack

- **Framework**: FastAPI
- **Speech Recognition**: Whisper (base model / Fine-tuned), PyTorch, HuggingFace Transformers
- **Audio Processing**: NumPy, LibROSA, wave, scipy
- **Machine Learning**: PyTorch, transformers pipeline
- **Containerization**: Docker, docker-compose
- **Development Environment**: Jupyter Notebook
- **Audio Formats**: WAV, MP3, PCM

## 📁 Project Structure

```
VoiceRecognitionAIAPI/
├── app/
│   ├── connectAPI/
│   │   ├── routers/
│   │   │   ├── __init__.py
│   │   │   ├── struct.py          # API structure definitions
│   │   │   └── stt.py             # STT endpoint implementation
│   │   └── service/
│   │       ├── errorcheck/
│   │       │   └── typeerror.py   # Type validation and error handling
│   │       ├── model/
│   │       │   ├── __init__.py
│   │       │   ├── models.py      # Model definitions and loading
│   │       │   └── trainedModel.pt # Fine-tuned model weights
│   │       ├── preprocessing/
│   │       │   ├── __init__.py
│   │       │   ├── mp3_to_wav.py  # Audio format conversion
│   │       │   ├── pcm_to_numpy.py # PCM processing
│   │       │   ├── processing.py   # Advanced audio preprocessing
│   │       │   └── wav_to_pcm.py  # WAV to PCM conversion
│   │       ├── __init__.py
│   │       └── control.py         # Main service controller
│   ├── temp/                      # Temporary file storage
│   ├── __init__.py
│   └── main.py                    # FastAPI application entry point
├── Managejupyter/
│   ├── __init__.py
│   ├── CheckTrained.ipynb         # Model training validation
│   └── temp/
├── venv/                          # Python virtual environment
├── temp/                          # Additional temporary storage
├── .dockerignore                  # Docker ignore patterns
├── .env                          # Environment variables
├── .gitignore                    # Git ignore patterns
├── docker-compose.yml            # Docker compose configuration
├── Dockerfile                    # Docker container definition
├── requirements.txt              # Python dependencies
├── result.json                   # Model evaluation results
├── server.log                    # Application logs
├── 000034.wav                    # Sample audio files
├── 000042.wav
└── 외부 라이브러리/               # External libraries
```

## 🚀 Features

### Core Functionality
- **Multi-format Audio Support**: WAV, MP3, PCM
- **Advanced Preprocessing**: Noise reduction, normalization, format conversion
- **Fine-tuned Models**: Custom trained models on Korean speech data
- **Error Handling**: Comprehensive type checking and validation
- **Performance Monitoring**: Detailed logging and evaluation metrics

### Audio Processing Pipeline
- **Format Conversion**: MP3 ↔ WAV ↔ PCM ↔ NumPy
- **Signal Processing**: Advanced audio enhancement and filtering
- **Feature Extraction**: MFCC and other acoustic features
- **Quality Validation**: Audio quality assessment and validation

## 📡 API Documentation

### Endpoints

#### POST /api/v1/stt
Speech-to-Text conversion endpoint with advanced preprocessing.

**Request:**
- Content-Type: `multipart/form-data`
- Body: audio file (`.wav`, `.mp3`, `.pcm`)
- Supported sample rates: 16kHz, 22kHz, 44.1kHz
- Max file size: 25MB

**Response:**
- Status: 200 OK
- Content-Type: `application/json`

```json
{
  "text": {
    "text": "text_data",
    "processing_method": "mfcc_simple"
  },
  "gender_result": [
    {
      "score": probability(float),
      "label": "female"
    },
    {
      "score": probability(float),
      "label": "male"
    }
  ]
}
```

**Error Response:**
```json
{
  "error": "Invalid audio format",
  "message": "Supported formats: wav, mp3, pcm",
  "code": 400
}
```

## 📊 Dataset

### KsponSpeech Dataset
- **Source**: AIHub Korea Sound Dataset
- **URL**: https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=123
- **Description**: Korean Free Speech Voice Data for Acoustic Models to Improve Conversational Speech Recognition Performance
- **Size**: 1000 hours of Korean conversation voices from 2,000 people
- **Environment**: Clean, quiet recording conditions
- **Content**: Two-person natural conversations on various topics

### Dataset Structure
```
KsponSpeech/
├── KsponSpeech_01.zip - 14.25GB ✅ (Primary training data)
├── KsponSpeech_02.zip - 14.26GB
├── KsponSpeech_03.zip - 14.18GB
├── KsponSpeech_04.zip - 14.23GB
└── KsponSpeech_05.zip - 14.57GB
```

## 🤖 Model Architecture

### Whisper-Small (Fine-tuned)
- **Base Model**: OpenAI Whisper (small)
- **Parameters**: 244M (240 million parameters)
- **Languages**: optimized for Korean
- **Learning Method**: fine-tuning by korean speaking data

### Model Selection Rationale
- **Performance vs. Size**: Optimal balance for production deployment
- **Korean Optimization**: Fine-tuned specifically for Korean speech patterns
- **Inference Speed**: Fast enough for real-time applications
- **Resource Efficiency**: Suitable for containerized microservices

### Fine-tuning Details
- **Training Data**: KsponSpeech Korean conversational speech
- **Training Duration**: Optimized for Korean phonetics
- **Model Weights**: `trainedModel.pt` (custom fine-tuned weights)

## 🔧 Development

### Local Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8081
```

## 🛠️ Audio Processing Features

### Advanced Preprocessing
- **Noise Reduction**: Spectral noise reduction algorithms
- **Normalization**: Dynamic range compression and amplitude normalization
- **Format Conversion**: Seamless conversion between audio formats
- **Quality Enhancement**: Audio quality improvement for better recognition

### Supported Features
- **Multi-channel Audio**: Mono/stereo support
- **Variable Sample Rates**: Automatic resampling
- **MFCC Feature Extraction**: Mel-frequency cepstral coefficients
- **Voice Activity Detection**: Intelligent speech segment detection

