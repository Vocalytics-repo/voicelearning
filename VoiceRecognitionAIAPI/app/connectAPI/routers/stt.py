from http.client import HTTPException

from fastapi import APIRouter, File, UploadFile
from typing import Dict, Any
from app.connectAPI.service.errorcheck.typeerror import errorhandling
from app.connectAPI.service.control import control_all
router = APIRouter()

@router.post("/api/v1/stt")
async def soundtotext(file: UploadFile = File(...)) -> Dict[str,Any]:
    if not errorhandling.soundcheck(file.filename):
        raise HTTPException(
            status_code = 400,
            detail ="파일 형식을 다시 체크하세요. .mp3 or .wav 파일만 허용됩니다."
        )
    else:
        file_content = await file.read()
        text = control_all(file_content)
        return text


