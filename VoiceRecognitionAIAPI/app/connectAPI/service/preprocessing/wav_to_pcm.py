from fastapi import UploadFile, HTTPException
import wave
import io
from typing import Union

async def convert_wav_to_pcm(wav_data: Union[bytes, UploadFile]) -> tuple[bytes, int]:
    """
    WAV 파일을 PCM 데이터로 변환 (개선된 버전)
    """
    try:
        # UploadFile 타입 처리
        if isinstance(wav_data, UploadFile):
            wav_bytes = await wav_data.read()
        else:
            wav_bytes = wav_data

        # BytesIO를 사용해 메모리에서 WAV 파일 처리
        with io.BytesIO(wav_bytes) as wav_buffer:
            with wave.open(wav_buffer, 'rb') as wav_file:
                # WAV 파일 파라미터 추출
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                framerate = wav_file.getframerate()
                n_frames = wav_file.getnframes()

                # PCM 데이터 읽기
                pcm_data = wav_file.readframes(n_frames)
                return pcm_data, sample_width

    except wave.Error as e:
        raise HTTPException(
            status_code=400,
            detail=f"잘못된 WAV 파일 형식: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"WAV -> PCM 변환 실패: {str(e)}"
        )

