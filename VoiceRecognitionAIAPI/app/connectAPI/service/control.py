from app.connectAPI.service.preprocessing.wav_to_pcm import convert_wav_to_pcm
from app.connectAPI.service.preprocessing.pcm_to_numpy import convert_pcm_to_numpy
from app.connectAPI.service.preprocessing.processing import convert_preprocessing
from app.connectAPI.service.model.models import sttmodel
from typing import Union, Dict, Any
from fastapi import UploadFile

async def control_all(wav_data : Union[bytes, UploadFile]) -> Dict[str,Any]:
    model = sttmodel()
    pcm_data, sample_width = await convert_wav_to_pcm(wav_data)
    numpy_data = await convert_pcm_to_numpy(pcm_data,sample_width)
    pre_pcm = await convert_preprocessing(numpy_data)
    text = model.start_stt(pre_pcm)
    return {"text":text}