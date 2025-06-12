import numpy as np
import librosa
import time


# =============================================================================
# 학습과 100% 동일한 전처리 파이프라인
# =============================================================================

async def convert_preprocessing(np_pcm, sample_rate=16000, use_simple=None, output_type='auto') -> np.ndarray:
    """
    학습과 100% 동일한 전처리 파이프라인
    - 학습 시 사용한 preprocess_and_save_pcm_features_only 함수와 동일한 로직
    """
    try:
        start_total = time.time()
        print(f"🔍 학습용 전처리 시작 - 데이터 길이: {len(np_pcm)}")

        # 입력 검증
        if len(np_pcm) == 0:
            print("❌ 빈 오디오 데이터")
            return np.zeros((39, 1))  # 기본 combined_features 형태

        # 🎯 학습과 100% 동일한 전처리 순서
        print("🔬 학습과 동일한 전처리 파이프라인 실행")

        # 1. 데이터가 이미 float32 형태라고 가정 (preprocessing에서 변환됨)
        audio_data = np_pcm
        print(f"  1️⃣ 입력 데이터 준비 완료")

        # 2. 프리엠퍼시스 적용 (학습과 동일)
        start = time.time()
        emphasized = apply_preemphasis(audio_data)
        print(f"  2️⃣ 프리엠퍼시스: {time.time() - start:.3f}초")

        # 3. 노이즈 제거 (학습과 동일)
        start = time.time()
        denoised = simple_noise_reduction(emphasized, sample_rate)
        print(f"  3️⃣ 노이즈 제거: {time.time() - start:.3f}초")

        # 4. 정규화 (학습과 동일: 'peak' 방식)
        start = time.time()
        normalized = normalize_audio(denoised, 'peak')
        print(f"  4️⃣ 정규화: {time.time() - start:.3f}초")

        # 5. VAD 적용 (학습과 동일)
        start = time.time()
        vad_mask, vad_frames = simple_vad(normalized, sample_rate)
        speech_only = normalized * vad_mask
        print(f"  5️⃣ VAD: {time.time() - start:.3f}초")

        # 6. MFCC 추출 (학습과 동일)
        start = time.time()
        mfcc_features = compute_mfcc(speech_only, sample_rate)
        print(f"  6️⃣ MFCC: {time.time() - start:.3f}초")

        # 7. 델타 및 델타-델타 계산 (학습과 동일)
        start = time.time()
        delta_features, delta2_features = compute_deltas(mfcc_features)
        print(f"  7️⃣ 델타: {time.time() - start:.3f}초")

        # 8. 모든 특성을 결합 (학습과 정확히 동일)
        start = time.time()
        combined_features = np.vstack([mfcc_features, delta_features, delta2_features])
        print(f"  8️⃣ 특성 결합: {time.time() - start:.3f}초")

        total_time = time.time() - start_total
        print(f"🎓 학습용 전처리 완료 - 총 시간: {total_time:.3f}초")
        print(f"📊 출력 형태: {combined_features.shape} (학습과 동일)")

        return combined_features

    except Exception as e:
        print(f"❌ 학습용 전처리 실패: {str(e)}")
        # 오류 시 기본 형태 반환
        return np.zeros((39, 100))


# =============================================================================
# 보조 함수들 (학습과 동일)
# =============================================================================

def apply_preemphasis(audio_data, coef=0.97):
    """Apply pre-emphasis filter to audio data"""
    return np.append(audio_data[0], audio_data[1:] - coef * audio_data[:-1])


def frame_signal(audio_data, sample_rate, frame_size=0.025, frame_stride=0.01, window='hamming'):
    """Divide the audio signal into overlapping frames and apply window function"""
    frame_length = int(frame_size * sample_rate)
    frame_step = int(frame_stride * sample_rate)

    signal_length = len(audio_data)
    num_frames = 1 + int(np.ceil((signal_length - frame_length) / frame_step))

    pad_length = (num_frames - 1) * frame_step + frame_length
    padded_signal = np.append(audio_data, np.zeros(pad_length - signal_length))

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = padded_signal[indices.astype(np.int32)]

    if window == 'hamming':
        frames = frames * np.hamming(frame_length)
    elif window == 'hanning':
        frames = frames * np.hanning(frame_length)

    return frames


def simple_noise_reduction(audio_data, sample_rate, noise_threshold=0.005):
    """Apply simple noise reduction by thresholding"""
    denoised = np.copy(audio_data)
    denoised[np.abs(denoised) < noise_threshold] = 0
    return denoised


def normalize_audio(audio_data, method='peak'):
    """Normalize audio data"""
    if method == 'peak':
        # 피크 기반 정규화 (-1 ~ 1 범위)
        if np.max(np.abs(audio_data)) > 0:
            return audio_data / np.max(np.abs(audio_data))
        else:
            return audio_data
    elif method == 'rms':
        # RMS 기반 정규화
        rms = np.sqrt(np.mean(audio_data ** 2))
        if rms > 0:
            return audio_data / (rms * 10)  # -0.1 ~ 0.1 범위
        else:
            return audio_data
    else:
        return audio_data


def simple_vad(audio_data, sample_rate, frame_size=0.025, frame_stride=0.01, energy_threshold=0.1):
    """Simple energy-based Voice Activity Detection (학습과 동일)"""
    try:
        # 프레임 길이와 간격을 샘플 단위로 변환
        frame_length = int(frame_size * sample_rate)
        frame_step = int(frame_stride * sample_rate)

        # 프레임 분할
        frames = frame_signal(audio_data, sample_rate, frame_size, frame_stride)

        # 각 프레임의 에너지 계산
        energy = np.sum(frames ** 2, axis=1)

        # 에너지 정규화
        if np.max(energy) > 0:
            energy = energy / np.max(energy)

        # 임계값 적용
        speech_frames = energy > energy_threshold

        # 프레임 단위 결정을 샘플 단위로 변환
        speech_mask = np.zeros_like(audio_data)

        for i, is_speech in enumerate(speech_frames):
            start_idx = i * frame_step
            end_idx = min(start_idx + frame_length, len(audio_data))
            speech_mask[start_idx:end_idx] = 1 if is_speech else 0

        # 정확히 원본 오디오 길이와 일치하도록 함
        speech_mask = speech_mask[:len(audio_data)]

        return speech_mask, speech_frames

    except Exception as e:
        print(f"⚠️ VAD 실패: {str(e)} - 전체를 음성으로 처리")
        return np.ones_like(audio_data), np.array([True])


def compute_mfcc(audio_data, sample_rate, num_cepstral=13):
    """Compute MFCC features from an audio signal using librosa (학습과 동일)"""
    try:
        mfcc_features = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=num_cepstral)
        return mfcc_features
    except Exception as e:
        print(f"⚠️ MFCC 계산 실패: {str(e)} - 더미 데이터 반환")
        return np.random.randn(num_cepstral, 100)


def compute_deltas(features):
    """Compute delta features using librosa (학습과 동일)"""
    try:
        delta_features = librosa.feature.delta(features)
        delta2_features = librosa.feature.delta(features, order=2)
        return delta_features, delta2_features
    except Exception as e:
        print(f"⚠️ 델타 계산 실패: {str(e)} - 제로 델타 반환")
        return np.zeros_like(features), np.zeros_like(features)


# =============================================================================
# 호환성을 위한 단순 버전 (필요시 사용)
# =============================================================================

async def convert_preprocessing_for_microphone_advanced(np_pcm, sample_rate=16000) -> np.ndarray:
    """마이크 입력용 고급 전처리 (VAD 포함)"""
    start = time.time()

    if len(np_pcm) == 0:
        return np.array([0.0])

    print("🎤 마이크 입력용 고급 전처리 시작")

    try:
        # 1. 프리엠퍼시스 적용
        emphasized = apply_preemphasis(np_pcm)

        # 2. 강화된 노이즈 제거
        denoised = simple_noise_reduction(emphasized, sample_rate, noise_threshold=0.01)

        # 3. 정규화
        normalized = normalize_audio(denoised, 'peak')

        # 4. VAD 적용 (음성 구간만 추출)
        vad_mask, vad_frames = simple_vad(normalized, sample_rate, energy_threshold=0.05)  # 임계값 낮춤
        speech_only = normalized * vad_mask

        # 5. 음성 구간이 너무 적으면 원본 사용
        speech_ratio = np.sum(vad_mask) / len(vad_mask)
        if speech_ratio < 0.1:  # 10% 미만이면
            print("  ⚠️ 음성 구간 부족 - 원본 사용")
            speech_only = normalized
        else:
            print(f"  ✅ 음성 구간 {speech_ratio * 100:.1f}% 추출")

        # 6. 볼륨 부스트
        boost_factor = 1.2
        speech_only = speech_only * boost_factor
        speech_only = np.clip(speech_only, -1.0, 1.0)

        # 7. 최종 정규화
        if np.max(np.abs(speech_only)) > 0:
            speech_only = speech_only / np.max(np.abs(speech_only))

        print(f"✅ 마이크용 고급 전처리: {time.time() - start:.3f}초")
        print(f"📊 출력: 1D 오디오 배열, 길이 {len(speech_only)}")

        return speech_only

    except Exception as e:
        print(f"❌ 고급 전처리 실패: {str(e)} - 기본 마이크 전처리로 fallback")
        return await convert_preprocessing_for_microphone_stable(np_pcm, sample_rate)


async def convert_preprocessing_for_microphone_stable(np_pcm, sample_rate=16000) -> np.ndarray:
    """마이크 입력용 안정적인 전처리 (단순하지만 효과적)"""
    start = time.time()

    if len(np_pcm) == 0:
        return np.array([0.0])

    print("🎤 마이크 입력용 안정적 전처리 시작")

    # 1. 기본 정규화
    if np.max(np.abs(np_pcm)) > 0:
        normalized = np_pcm / np.max(np.abs(np_pcm))
    else:
        normalized = np_pcm

    # 2. 적당한 노이즈 제거 (너무 강하지 않게)
    noise_threshold = 0.015  # 0.01과 0.02 사이
    normalized[np.abs(normalized) < noise_threshold] = 0

    # 3. 볼륨 부스트 (적당히)
    boost_factor = 1.3
    normalized = normalized * boost_factor
    normalized = np.clip(normalized, -1.0, 1.0)

    # 4. 최종 정규화
    if np.max(np.abs(normalized)) > 0:
        normalized = normalized / np.max(np.abs(normalized))

    print(f"✅ 안정적 마이크 전처리: {time.time() - start:.3f}초")
    print(f"📊 출력: 1D 오디오 배열, 길이 {len(normalized)}")

    return normalized


async def convert_preprocessing_for_microphone_chunked(np_pcm, sample_rate=16000, max_chunk_duration=10) -> np.ndarray:
    """마이크 입력용 청크 분할 전처리 (긴 오디오 대응)"""
    start = time.time()

    if len(np_pcm) == 0:
        return np.array([0.0])

    # 오디오 길이 계산 (초)
    duration = len(np_pcm) / sample_rate

    print(f"🎤 마이크 입력 분석: {duration:.1f}초")

    # 짧은 오디오는 기존 방식 사용
    if duration <= max_chunk_duration:
        return await convert_preprocessing_for_microphone_advanced(np_pcm, sample_rate)

    print(f"⚠️ 긴 오디오 감지 ({duration:.1f}초) - 청크 분할 처리")

    # 청크 크기 계산
    chunk_size = int(max_chunk_duration * sample_rate)
    chunks = []

    # 오디오를 청크로 분할
    for i in range(0, len(np_pcm), chunk_size):
        chunk = np_pcm[i:i + chunk_size]

        # 각 청크에 고급 전처리 적용
        processed_chunk = await convert_preprocessing_for_microphone_advanced(chunk, sample_rate)
        chunks.append(processed_chunk)

    # 청크들을 합치기
    result = np.concatenate(chunks)

    print(f"✅ 청크 분할 전처리 완료: {time.time() - start:.3f}초")
    print(f"📊 출력: 1D 오디오 배열, 길이 {len(result)} ({len(chunks)}개 청크)")

    return result


async def convert_preprocessing_simple(np_pcm) -> np.ndarray:
    """기존 단순 정규화 버전 (호환성용) - 1D 오디오 출력"""
    start = time.time()

    if len(np_pcm) == 0:
        return np.array([0.0])

    # 단순 정규화만 수행
    if np.max(np.abs(np_pcm)) > 0:
        normalized = np_pcm / np.max(np.abs(np_pcm))
    else:
        normalized = np_pcm

    # 간단한 노이즈 제거 (선택적)
    noise_threshold = 0.01
    normalized[np.abs(normalized) < noise_threshold] = 0

    print(f"✅ 단순 전처리: {time.time() - start:.3f}초")
    print(f"📊 출력: 1D 오디오 배열, 길이 {len(normalized)}")

    return normalized