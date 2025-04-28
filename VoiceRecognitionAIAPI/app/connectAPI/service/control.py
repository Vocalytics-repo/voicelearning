from app.connectAPI.service.preprocessing.wav_to_pcm import convert_wav_to_pcm
from app.connectAPI.service.preprocessing.pcm_to_numpy import convert_pcm_to_numpy
from app.connectAPI.service.preprocessing.processing import convert_preprocessing
from app.connectAPI.service.model.models import sttmodel
from typing import Union, Dict, Any
from fastapi import UploadFile

async def control_all(wav_data : Union[bytes, UploadFile]) -> Dict[str,Any]:
    pcm_data, sample_width = convert_wav_to_pcm(wav_data)
    numpy_data = convert_pcm_to_numpy(pcm_data,sample_width)
    pre_pcm = convert_preprocessing(numpy_data)
    text = sttmodel.start_stt(pre_pcm)
    return {"text":text}