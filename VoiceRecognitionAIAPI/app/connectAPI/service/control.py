from app.connectAPI.service.preprocessing.wav_to_pcm import convert_wav_to_pcm
from app.connectAPI.service.preprocessing.pcm_to_numpy import convert_pcm_to_numpy
from app.connectAPI.service.preprocessing.processing import (
    convert_preprocessing,
    convert_preprocessing_for_microphone_stable,
    convert_preprocessing_for_microphone_advanced,
    convert_preprocessing_for_microphone_chunked,
    convert_preprocessing_simple,
    intelligent_feature_processing,
    detect_feature_type
)
import numpy as np
from app.connectAPI.service.model.models import get_stt_model
from typing import Union, Dict, Any
from fastapi import UploadFile
import wave
import io
import librosa


async def control_all(wav_data: Union[bytes, UploadFile]) -> Dict[str, Any]:
    """
    통합 오디오 처리 및 음성 인식 함수
    stt.py와 완전 호환 - 자동으로 최적 처리 방법 선택
    """
    model = get_stt_model()

    try:
        # WAV 데이터 준비
        if isinstance(wav_data, UploadFile):
            wav_content = await wav_data.read()
            await wav_data.seek(0)
        else:
            wav_content = wav_data

        # 원본 WAV 파일 정보 분석
        audio_info = await _analyze_wav_info(wav_content)
        print(f"📊 원본 WAV 정보: {audio_info}")

        # 샘플링 레이트에 따른 처리 분기
        if audio_info['sample_rate'] != 16000:
            # 16kHz가 아닌 경우: LibROSA로 변환 후 간단 처리
            return await _handle_non_16khz_audio(wav_content, model, audio_info)
        else:
            # 16kHz인 경우: 기존 파이프라인 사용
            return await _handle_16khz_audio(wav_content, model, audio_info)

    except Exception as e:
        print(f"❌ 전체 처리 실패: {e}")
        # 폴백: LibROSA로 최소 처리
        return await _fallback_processing(wav_data, model)


async def _handle_non_16khz_audio(wav_content: bytes, model, audio_info: Dict[str, Any]) -> Dict[str, Any]:
    """16kHz가 아닌 오디오 처리"""
    try:
        print(f"🔄 샘플링 레이트 변환: {audio_info['sample_rate']}Hz → 16000Hz")

        # LibROSA로 16kHz 변환 및 최소 전처리
        audio_data, _ = librosa.load(io.BytesIO(wav_content), sr=16000)
        processed_audio = await _optimized_preprocessing(audio_data)

        # 음성 인식
        text = model.transcribe(processed_audio)

        return {
            "text": text,
            "processing_method": f"resampled_from_{audio_info['sample_rate']}Hz"
        }

    except Exception as e:
        print(f"❌ 리샘플링 처리 실패: {e}")
        raise


async def _handle_16khz_audio(wav_content: bytes, model, audio_info: Dict[str, Any]) -> Dict[str, Any]:
    """16kHz 오디오 처리 - 기존 파이프라인 우선"""
    try:
        print("🎯 16kHz 오디오 처리 시작")

        # 기존 파이프라인으로 데이터 변환
        pcm_data, sample_width = await _convert_wav_to_pcm_bytes(wav_content)
        numpy_data = await convert_pcm_to_numpy(pcm_data, sample_width)

        # 오디오 분석
        duration = len(numpy_data) / 16000
        audio_max = np.max(np.abs(numpy_data))
        audio_rms = np.sqrt(np.mean(numpy_data ** 2))

        print(f"🔍 오디오 분석: 길이 {duration:.1f}초, 최대값 {audio_max:.4f}, RMS {audio_rms:.4f}")

        # 길이와 품질에 따른 전처리 선택
        processed_audio = await _select_preprocessing_method(numpy_data, duration, audio_max, audio_rms)

        # 음성 인식
        text = model.transcribe(processed_audio)

        return {"text": text, "processing_method": "legacy_pipeline"}

    except Exception as e:
        print(f"❌ 16kHz 처리 실패: {e}")
        raise


async def _select_preprocessing_method(
        numpy_data: np.ndarray,
        duration: float,
        max_amplitude: float,
        rms: float
) -> np.ndarray:
    """오디오 특성에 따른 최적 전처리 방법 선택"""

    try:
        print(f"🎛️ 전처리 방법 선택 중...")

        # 1. 매우 짧은 오디오
        if duration < 1.0:
            print("⚡ 짧은 오디오 - 간단 전처리")
            return await convert_preprocessing_simple(numpy_data)

        # 2. 매우 긴 오디오
        elif duration > 30.0:
            print("📊 긴 오디오 - 청킹 전처리")
            return await convert_preprocessing_for_microphone_chunked(numpy_data)

        # 3. 중간 길이 오디오 (대부분의 케이스)
        elif duration < 8.0:
            print("🎤 중간 길이 - 안정화 전처리")
            return await convert_preprocessing_for_microphone_stable(numpy_data)

        # 4. 품질에 따른 선택 (8초 이상)
        else:
            # 정규화 상태 확인
            is_normalized = max_amplitude <= 1.0

            # 고품질 오디오
            if is_normalized and 0.03 <= rms <= 0.5:
                print("🎵 고품질 오디오 - 기본 전처리")
                return await convert_preprocessing(numpy_data)

            # 저품질 또는 비정규화 오디오
            elif not is_normalized or rms < 0.03 or max_amplitude > 0.98:
                print("🔧 저품질/비정규화 오디오 - 고급 전처리")
                return await convert_preprocessing_for_microphone_advanced(numpy_data)

            # 기본값
            else:
                print("🎤 일반 오디오 - 안정화 전처리")
                return await convert_preprocessing_for_microphone_stable(numpy_data)

    except Exception as e:
        print(f"⚠️ 전처리 선택 실패: {e} - 간단 전처리로 폴백")
        return await convert_preprocessing_simple(numpy_data)


async def _optimized_preprocessing(audio_data: np.ndarray) -> np.ndarray:
    """
    리샘플링된 오디오용 최적화된 전처리 (노이즈 게이트 개선)
    """
    try:
        print("🎵 최적화된 전처리 적용")

        # 1. DC 성분 제거
        audio_data = audio_data - np.mean(audio_data)

        # 2. 정규화
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val

        # 3. 매우 낮은 노이즈 게이트 (기존 0.005 → 0.0005로 개선)
        noise_threshold = 0.0005
        audio_data = np.where(np.abs(audio_data) < noise_threshold, 0, audio_data)

        # 4. 최소한의 페이드 (기존 //20 → //50으로 줄임)
        window_size = min(256, len(audio_data) // 50)
        if window_size > 0:
            fade_in = np.linspace(0, 1, window_size)
            fade_out = np.linspace(1, 0, window_size)
            audio_data[:window_size] *= fade_in
            audio_data[-window_size:] *= fade_out

        print(f"✅ 최적화된 전처리 완료: 최대값 {np.max(np.abs(audio_data)):.4f}")
        return audio_data

    except Exception as e:
        print(f"❌ 최적화된 전처리 실패: {e}")
        return audio_data


async def _convert_wav_to_pcm_bytes(wav_content: bytes) -> tuple:
    """WAV 바이트를 PCM 데이터로 변환"""
    try:
        print("🔄 WAV → PCM 변환")

        with wave.open(io.BytesIO(wav_content), 'rb') as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()

            print(f"📊 변환 완료: SR={sample_rate}Hz, width={sample_width}bytes")
            return frames, sample_width

    except Exception as direct_error:
        print(f"⚠️ 직접 변환 실패: {direct_error}")

        # 폴백: 기존 함수 호환 래퍼
        try:
            print("🔄 호환 래퍼로 재시도")

            class BytesUploadFile:
                def __init__(self, content: bytes):
                    self._content = content
                    self._position = 0

                async def read(self, size: int = -1):
                    if size == -1:
                        result = self._content[self._position:]
                        self._position = len(self._content)
                    else:
                        result = self._content[self._position:self._position + size]
                        self._position += len(result)
                    return result

                async def seek(self, position: int):
                    self._position = position

                @property
                def filename(self):
                    return "audio.wav"

                @property
                def content_type(self):
                    return "audio/wav"

            fake_file = BytesUploadFile(wav_content)
            return await convert_wav_to_pcm(fake_file)

        except Exception as wrapper_error:
            print(f"❌ 래퍼 변환도 실패: {wrapper_error}")
            raise Exception(f"WAV→PCM 변환 실패: {direct_error}")


async def _analyze_wav_info(wav_content: bytes) -> Dict[str, Any]:
    """WAV 파일 정보 분석"""
    try:
        with wave.open(io.BytesIO(wav_content), 'rb') as wav_file:
            info = {
                'sample_rate': wav_file.getframerate(),
                'channels': wav_file.getnchannels(),
                'sample_width': wav_file.getsampwidth(),
                'frames': wav_file.getnframes(),
                'duration': wav_file.getnframes() / wav_file.getframerate()
            }
            return info
    except Exception as e:
        print(f"⚠️ WAV 정보 분석 실패: {e}")
        return {
            'sample_rate': 16000,
            'channels': 1,
            'sample_width': 2,
            'frames': 0,
            'duration': 0
        }


async def _fallback_processing(wav_data: Union[bytes, UploadFile], model) -> Dict[str, Any]:
    """최후의 폴백 처리"""
    try:
        print("🆘 폴백 처리 시작")

        # WAV 데이터 준비
        if isinstance(wav_data, UploadFile):
            wav_content = await wav_data.read()
        else:
            wav_content = wav_data

        # LibROSA로 최소 처리
        audio_data, _ = librosa.load(io.BytesIO(wav_content), sr=16000)

        # 기본 정규화만
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))

        print(f"✅ 폴백 처리 완료: {len(audio_data)} samples")

        # 음성 인식
        text = model.transcribe(audio_data)
        return {"text": text, "processing_method": "fallback"}

    except Exception as e:
        print(f"❌ 폴백 처리 실패: {e}")
        return {"text": "음성 인식 실패"}


# =============================================================================
# 추가 처리 모드들 (선택적 사용)
# =============================================================================

async def simple_control_all(wav_data: Union[bytes, UploadFile]) -> Dict[str, Any]:
    """간단한 처리 모드 - 무조건 convert_preprocessing_simple 사용"""
    try:
        print("⚡ 간단 처리 모드")

        # 기존 파이프라인으로 처리
        pcm_data, sample_width = await convert_wav_to_pcm(wav_data)
        numpy_data = await convert_pcm_to_numpy(pcm_data, sample_width)

        # 간단 전처리만
        processed_audio = await convert_preprocessing_simple(numpy_data)

        # 음성 인식
        model = get_stt_model()
        text = model.transcribe(processed_audio)

        return {"text": text, "processing_method": "simple"}

    except Exception as e:
        print(f"❌ 간단 처리 실패: {e}")
        return {"text": "간단 처리 실패"}


async def mfcc_control_all(wav_data: Union[bytes, UploadFile]) -> Dict[str, Any]:
    """MFCC 특성 기반 처리 모드"""
    try:
        print("🎓 MFCC 특성 처리 모드")

        # WAV 데이터 준비
        if isinstance(wav_data, UploadFile):
            wav_content = await wav_data.read()
        else:
            wav_content = wav_data

        # LibROSA로 로드
        audio_data, _ = librosa.load(io.BytesIO(wav_content), sr=16000)

        # 2D 배열인지 확인 (MFCC 특성 데이터)
        if len(audio_data.shape) == 2:
            feature_info = detect_feature_type(audio_data)
            if feature_info['confidence'] > 0.7:
                processed_audio = await intelligent_feature_processing(audio_data, 16000)
                model = get_stt_model()
                text = model.transcribe(processed_audio)
                return {"text": text, "processing_method": "mfcc_features"}

        # 일반 오디오인 경우 간단 처리
        processed_audio = await convert_preprocessing_simple(audio_data)
        model = get_stt_model()
        text = model.transcribe(processed_audio)
        return {"text": text, "processing_method": "mfcc_simple"}

    except Exception as e:
        print(f"❌ MFCC 처리 실패: {e}")
        return {"text": "MFCC 처리 실패"}


# =============================================================================
# 함수 별칭 (하위 호환성)
# =============================================================================

# 메인 함수 (stt.py에서 사용)
process_audio = control_all

# 추가 모드들
simple_process_audio = simple_control_all
mfcc_process_audio = mfcc_control_all