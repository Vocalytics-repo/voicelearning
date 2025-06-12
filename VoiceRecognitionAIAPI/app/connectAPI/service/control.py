from app.connectAPI.service.preprocessing.wav_to_pcm import convert_wav_to_pcm
from app.connectAPI.service.preprocessing.pcm_to_numpy import convert_pcm_to_numpy
from app.connectAPI.service.preprocessing.processing import convert_preprocessing,convert_preprocessing_for_microphone_stable,convert_preprocessing_for_microphone_advanced,convert_preprocessing_for_microphone_chunked,convert_preprocessing_simple
import numpy as np
from app.connectAPI.service.model.models import get_stt_model
from typing import Union, Dict, Any
from fastapi import UploadFile


async def control_all(wav_data: Union[bytes, UploadFile]) -> Dict[str, Any]:
    model = get_stt_model()

    # 🔍 원본 WAV 파일 샘플링 레이트 확인
    try:
        import wave
        import io
        import librosa

        if isinstance(wav_data, UploadFile):
            wav_content = await wav_data.read()
            await wav_data.seek(0)
        else:
            wav_content = wav_data

        with wave.open(io.BytesIO(wav_content), 'rb') as wav_file:
            original_sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            duration = wav_file.getnframes() / original_sample_rate

        print(f"📊 원본 WAV 정보:")
        print(f"  - 샘플링 레이트: {original_sample_rate} Hz")
        print(f"  - 채널 수: {channels}")
        print(f"  - 재생 시간: {duration:.2f}초")

        # 🎯 24kHz 오디오 처리
        if original_sample_rate != 16000:
            print(f"🔧 리샘플링 필요: {original_sample_rate}Hz → 16000Hz")

            # librosa로 직접 16kHz로 로드
            audio_data, _ = librosa.load(io.BytesIO(wav_content), sr=16000)

            # 정규화
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))

            print(f"✅ 리샘플링 완료: {len(audio_data)} samples, {len(audio_data) / 16000:.2f}초")

            # 🎯 전처리 스킵하고 바로 사용
            text = model.transcribe(audio_data)
            return {"text": text}

    except Exception as e:
        print(f"⚠️ WAV 정보 확인 실패: {e}")

    # 기존 처리 (16kHz인 경우)
    pcm_data, sample_width = await convert_wav_to_pcm(wav_data)
    numpy_data = await convert_pcm_to_numpy(pcm_data, sample_width)

    audio_duration = len(numpy_data) / 16000
    audio_max = np.max(np.abs(numpy_data))
    print(f"🔍 전처리 후 분석: 길이 {audio_duration:.1f}초, 최대값 {audio_max:.4f}")

    pre_pcm = await convert_preprocessing_simple(numpy_data)
    text = model.transcribe(pre_pcm)
    return {"text": text}