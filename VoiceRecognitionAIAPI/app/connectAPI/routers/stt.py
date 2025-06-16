from http.client import HTTPException
from fastapi import APIRouter, File, UploadFile
from typing import Dict, Any
from fastapi.responses import JSONResponse
from app.connectAPI.service.errorcheck.typeerror import errorhandling
from app.connectAPI.service.control import control_all, mfcc_control_all
from app.connectAPI.service.preprocessing.mp3_to_wav import mp3_to_wav
from transformers import pipeline

# 사용
router = APIRouter()

@router.post("/api/v1/stt")
async def soundtotext(file: UploadFile = File(...)) -> Dict[str,Any]:
    '''
    STT API에 접근하기 위한 함수
    1. .mp3, .wav 확장자 타입의 파일을 받지만, 결과적으로 처리 자체는 .wav파일로 변환하여 처리
    2. 해당 API를 통하여 구현하여 router에는 해당 함수만 존재
    3. 구체적인 기능적인 코드는 service 디렉토리에 들어 있음
    4. v1의 경우 단일 컴포넌트(서버)로 구성됨
    5. 시간에 따라 v2 (model - 컴포넌트 / 데이터 전처리 - 컴포넌트) 2개의 컴포넌트(서버)로 구성할 수 있음
    6. 추가적으로 모델 컴포넌트를 GPU서버에 구현하는 것으로 구동 방법도 생각 중
    내부 동작을 통하여 text를 반환함.
    '''
    error_handler = errorhandling()
    if not error_handler.soundcheck(file.filename):
        raise HTTPException(
            status_code = 400,
            detail ="파일 형식을 다시 체크하세요. .mp3 or .wav 파일만 허용됩니다."
        )
    else:
        file_content = await file.read()
        wav_file_data = await mp3_to_wav(file_content, file.filename)
        #text = await control_all(wav_file_data)
        text = await mfcc_control_all(wav_file_data)
        gender_classifier = pipeline("audio-classification",
                                     model="alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech")
        result = gender_classifier(wav_file_data)
        response_data = {
            "text" : text,
            "gender_result": result
        }
        return JSONResponse(content=response_data)


