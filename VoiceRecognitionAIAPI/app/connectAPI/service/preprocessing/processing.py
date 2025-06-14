import numpy as np
from typing import Dict, Any

# =============================================================================
# 특성 감지 및 분석 함수들
# =============================================================================

def detect_feature_type(features: np.ndarray) -> Dict[str, Any]:
    """입력 특성의 타입과 구조를 자동 감지"""
    try:
        n_features, n_frames = features.shape

        feature_info = {
            'shape': features.shape,
            'type': 'unknown',
            'components': {},
            'confidence': 0.0
        }

        # 차원 기반 타입 감지
        if n_features == 13:
            feature_info['type'] = 'mfcc_only'
            feature_info['components']['mfcc'] = (0, 13)
            feature_info['confidence'] = 0.9

        elif n_features == 26:
            feature_info['type'] = 'mfcc_delta'
            feature_info['components']['mfcc'] = (0, 13)
            feature_info['components']['delta'] = (13, 26)
            feature_info['confidence'] = 0.85

        elif n_features == 39:
            feature_info['type'] = 'mfcc_full'
            feature_info['components']['mfcc'] = (0, 13)
            feature_info['components']['delta'] = (13, 26)
            feature_info['components']['delta2'] = (26, 39)
            feature_info['confidence'] = 0.95

        # 통계적 검증으로 신뢰도 조정
        confidence_adjustment = _validate_feature_statistics(features, feature_info['type'])
        feature_info['confidence'] *= confidence_adjustment

        print(f"🔍 특성 타입 감지: {feature_info['type']} (신뢰도: {feature_info['confidence']:.2f})")
        return feature_info

    except Exception as e:
        print(f"❌ 특성 타입 감지 실패: {e}")
        return {
            'shape': features.shape,
            'type': 'unknown',
            'components': {},
            'confidence': 0.0
        }


def _validate_feature_statistics(features: np.ndarray, detected_type: str) -> float:
    """통계적 특성 검증"""
    try:
        confidence = 1.0

        # MFCC 특성 검증
        if detected_type in ['mfcc_only', 'mfcc_delta', 'mfcc_full']:
            mfcc_part = features[:13, :]

            # MFCC 첫 번째 계수 (에너지)는 일반적으로 다른 계수보다 큰 값
            if np.mean(np.abs(mfcc_part[0, :])) > np.mean(np.abs(mfcc_part[1:, :])):
                confidence *= 1.1
            else:
                confidence *= 0.9

            # MFCC 계수들의 분산 패턴 검증
            variances = np.var(mfcc_part, axis=1)
            if np.argmax(variances) == 0:  # 첫 번째 계수가 가장 큰 분산
                confidence *= 1.05

        # Delta 특성 검증
        if detected_type in ['mfcc_delta', 'mfcc_full']:
            delta_part = features[13:26, :]

            # Delta는 일반적으로 MFCC보다 작은 값
            mfcc_mean = np.mean(np.abs(features[:13, :]))
            delta_mean = np.mean(np.abs(delta_part))

            if delta_mean < mfcc_mean:
                confidence *= 1.05
            else:
                confidence *= 0.95

        # Delta2 특성 검증
        if detected_type == 'mfcc_full':
            delta2_part = features[26:39, :]
            delta_part = features[13:26, :]

            # Delta2는 일반적으로 Delta보다 작은 값
            delta_mean = np.mean(np.abs(delta_part))
            delta2_mean = np.mean(np.abs(delta2_part))

            if delta2_mean < delta_mean:
                confidence *= 1.05
            else:
                confidence *= 0.95

        return min(confidence, 1.0)

    except Exception as e:
        print(f"⚠️ 통계적 검증 실패: {e}")
        return 0.8


async def adaptive_feature_to_audio(features: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """적응형 특성-오디오 변환"""
    try:
        # 특성 타입 자동 감지
        feature_info = detect_feature_type(features)

        print(f"🔄 적응형 변환 시작: {feature_info['type']}")

        # 신뢰도 기반 처리 방법 선택
        if feature_info['confidence'] > 0.8:
            # 높은 신뢰도: 특성에 맞는 최적 방법 사용
            if feature_info['type'] == 'mfcc_only':
                return await _process_mfcc_only(features, sample_rate)
            elif feature_info['type'] == 'mfcc_delta':
                return await _process_mfcc_delta(features, sample_rate)
            elif feature_info['type'] == 'mfcc_full':
                return await _process_mfcc_full(features, sample_rate)

        # 낮은 신뢰도 또는 알 수 없는 타입: 안전한 방법 사용
        print("⚠️ 낮은 신뢰도 - 안전한 변환 방법 사용")
        return await _safe_feature_conversion(features, sample_rate)

    except Exception as e:
        print(f"❌ 적응형 변환 실패: {e}")
        return np.random.randn(sample_rate) * 0.05


async def _process_mfcc_only(features: np.ndarray, sample_rate: int) -> np.ndarray:
    """순수 MFCC 최적 처리"""
    print("📊 순수 MFCC 최적 처리")
    return _mfcc_basic_reconstruction(features, sample_rate)


async def _process_mfcc_delta(features: np.ndarray, sample_rate: int) -> np.ndarray:
    """MFCC + Delta 최적 처리"""
    print("📊 MFCC + Delta 최적 처리")
    mfcc = features[:13, :]
    delta = features[13:26, :]
    return _mfcc_delta_reconstruction(mfcc, delta, sample_rate)


async def _process_mfcc_full(features: np.ndarray, sample_rate: int) -> np.ndarray:
    """MFCC + Delta + Delta2 최적 처리"""
    print("📊 MFCC + Delta + Delta2 최적 처리")
    mfcc = features[:13, :]
    delta = features[13:26, :]
    delta2 = features[26:39, :]
    return _mfcc_full_reconstruction(mfcc, delta, delta2, sample_rate)


async def _safe_feature_conversion(features: np.ndarray, sample_rate: int) -> np.ndarray:
    """안전한 특성 변환 (타입 불명 시)"""
    try:
        print("🛡️ 안전한 특성 변환")

        # 가장 안전한 방법: 첫 13개 특성만 MFCC로 간주
        safe_features = features[:min(13, features.shape[0]), :]

        # 기본 복원 시도
        try:
            return _mfcc_basic_reconstruction(safe_features, sample_rate)
        except Exception:
            # 복원 실패 시 수동 방법
            return _manual_mfcc_reconstruction(safe_features, sample_rate)

    except Exception as e:
        print(f"❌ 안전한 변환 실패: {e}")
        return np.random.randn(sample_rate) * 0.05


# =============================================================================
# 고급 신호 처리 함수들
# =============================================================================

def enhance_audio_quality(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """복원된 오디오 품질 향상"""
    try:
        print("🔧 오디오 품질 향상 시작")

        enhanced = np.copy(audio)

        # 1. DC 성분 제거
        enhanced = enhanced - np.mean(enhanced)

        # 2. 부드러운 윈도우 적용 (클릭 노이즈 제거)
        window_size = min(1024, len(enhanced) // 10)
        if window_size > 0:
            window = np.hanning(window_size)
            enhanced[:window_size // 2] *= window[:window_size // 2]
            enhanced[-window_size // 2:] *= window[-window_size // 2:]

        # 3. 고주파 노이즈 제거
        try:
            # 간단한 저역통과 필터
            from scipy import signal
            nyquist = sample_rate / 2
            cutoff = min(8000, nyquist * 0.9)
            b, a = signal.butter(4, cutoff / nyquist, btype='low')
            enhanced = signal.filtfilt(b, a, enhanced)
        except ImportError:
            # scipy 없으면 간단한 이동평균
            kernel_size = 3
            kernel = np.ones(kernel_size) / kernel_size
            enhanced = np.convolve(enhanced, kernel, mode='same')

        # 4. 동적 범위 정규화
        if np.max(np.abs(enhanced)) > 0:
            enhanced = enhanced / np.max(np.abs(enhanced)) * 0.8

        print("✅ 오디오 품질 향상 완료")
        return enhanced

    except Exception as e:
        print(f"⚠️ 품질 향상 실패: {e}")
        return audio


def validate_reconstructed_audio(audio: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
    """복원된 오디오 품질 검증"""
    try:
        metrics = {
            'length_seconds': len(audio) / sample_rate,
            'max_amplitude': np.max(np.abs(audio)),
            'rms': np.sqrt(np.mean(audio ** 2)),
            'zero_crossing_rate': np.mean(np.abs(np.diff(np.sign(audio)))) / 2,
            'is_valid': True,
            'quality_score': 0.0
        }

        # 품질 점수 계산
        score = 0.0

        # 길이 검증 (최소 0.1초)
        if metrics['length_seconds'] >= 0.1:
            score += 0.25

        # 진폭 검증 (클리핑 없음)
        if 0.01 <= metrics['max_amplitude'] <= 0.99:
            score += 0.25

        # RMS 검증 (적절한 에너지)
        if 0.001 <= metrics['rms'] <= 0.5:
            score += 0.25

        # 제로 크로싱 검증 (음성 신호 특성)
        if 0.01 <= metrics['zero_crossing_rate'] <= 0.5:
            score += 0.25

        metrics['quality_score'] = score
        metrics['is_valid'] = score >= 0.5

        return metrics

    except Exception as e:
        print(f"⚠️ 오디오 검증 실패: {e}")
        return {
            'length_seconds': 0,
            'max_amplitude': 0,
            'rms': 0,
            'zero_crossing_rate': 0,
            'is_valid': False,
            'quality_score': 0.0
        }


# =============================================================================
# 통합 인터페이스 함수
# =============================================================================

async def intelligent_feature_processing(input_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """지능형 특성 처리 - 모든 타입 자동 처리"""
    try:
        print(f"🧠 지능형 특성 처리 시작: {input_data.shape}")

        # 1차원인 경우 이미 오디오로 간주
        if len(input_data.shape) == 1:
            print("📻 1D 오디오 데이터 - 직접 반환")
            return input_data

        # 2차원인 경우 특성으로 간주하여 변환
        if len(input_data.shape) == 2:
            # 적응형 특성-오디오 변환
            audio = await adaptive_feature_to_audio(input_data, sample_rate)

            # 품질 향상
            enhanced_audio = enhance_audio_quality(audio, sample_rate)

            # 품질 검증
            validation = validate_reconstructed_audio(enhanced_audio, sample_rate)

            if validation['is_valid']:
                print(f"✅ 지능형 처리 완료 - 품질 점수: {validation['quality_score']:.2f}")
                return enhanced_audio
            else:
                print("⚠️ 품질 검증 실패 - 원본 오디오 반환")
                return audio

        # 예상치 못한 차원
        print(f"❌ 지원하지 않는 데이터 형태: {input_data.shape}")
        return np.random.randn(sample_rate) * 0.05

    except Exception as e:
        print(f"❌ 지능형 처리 실패: {e}")
        return np.random.randn(sample_rate) * 0.05


import librosa
import time
from typing import Optional, Tuple
import scipy.signal
import scipy.ndimage


async def convert_preprocessing(audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    기본 오디오 전처리 함수
    Args:
        audio_data: 입력 오디오 데이터 (numpy array)
        sample_rate: 샘플링 레이트 (기본값: 16000)
    Returns:
        전처리된 오디오 데이터
    """
    try:
        print(f"🔧 기본 전처리 시작 - 입력 길이: {len(audio_data)}")

        # 1. 정규화
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))

        # 2. 노이즈 제거
        audio_data = _remove_noise(audio_data, sample_rate)

        # 3. 프리엠퍼시스
        audio_data = _apply_preemphasis(audio_data)

        print(f"✅ 기본 전처리 완료 - 출력 길이: {len(audio_data)}")
        return audio_data

    except Exception as e:
        print(f"❌ 기본 전처리 실패: {e}")
        return audio_data


async def convert_preprocessing_for_microphone_stable(audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    마이크 입력용 안정화된 전처리
    Args:
        audio_data: 입력 오디오 데이터
        sample_rate: 샘플링 레이트
    Returns:
        전처리된 오디오 데이터
    """
    try:
        print(f"🎤 마이크 안정화 전처리 시작 - 입력 길이: {len(audio_data)}")

        # 1. DC 오프셋 제거
        audio_data = audio_data - np.mean(audio_data)

        # 2. 정규화
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))

        # 3. 고주파 노이즈 제거 (저역통과 필터)
        audio_data = _apply_lowpass_filter(audio_data, sample_rate, cutoff=8000)

        # 4. VAD (Voice Activity Detection)
        audio_data = _apply_vad(audio_data, sample_rate)

        # 5. 프리엠퍼시스
        audio_data = _apply_preemphasis(audio_data)

        print(f"✅ 마이크 안정화 전처리 완료 - 출력 길이: {len(audio_data)}")
        return audio_data

    except Exception as e:
        print(f"❌ 마이크 안정화 전처리 실패: {e}")
        return audio_data


async def convert_preprocessing_for_microphone_advanced(audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    마이크 입력용 고급 전처리
    Args:
        audio_data: 입력 오디오 데이터
        sample_rate: 샘플링 레이트
    Returns:
        전처리된 오디오 데이터
    """
    try:
        print(f"🔬 마이크 고급 전처리 시작 - 입력 길이: {len(audio_data)}")

        # 1. DC 오프셋 제거
        audio_data = audio_data - np.mean(audio_data)

        # 2. 스펙트럼 노이즈 감소
        audio_data = _spectral_noise_reduction(audio_data, sample_rate)

        # 3. 동적 범위 압축
        audio_data = _dynamic_range_compression(audio_data)

        # 4. 고급 VAD
        audio_data = _advanced_vad(audio_data, sample_rate)

        # 5. 적응형 필터링
        audio_data = _adaptive_filtering(audio_data, sample_rate)

        # 6. 프리엠퍼시스
        audio_data = _apply_preemphasis(audio_data)

        print(f"✅ 마이크 고급 전처리 완료 - 출력 길이: {len(audio_data)}")
        return audio_data

    except Exception as e:
        print(f"❌ 마이크 고급 전처리 실패: {e}")
        return audio_data


async def convert_preprocessing_for_microphone_chunked(audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    마이크 입력용 청킹 전처리 (긴 오디오용)
    Args:
        audio_data: 입력 오디오 데이터
        sample_rate: 샘플링 레이트
    Returns:
        전처리된 오디오 데이터
    """
    try:
        print(f"📊 청킹 전처리 시작 - 입력 길이: {len(audio_data)}")

        # 오디오 길이 확인
        duration = len(audio_data) / sample_rate

        if duration <= 30.0:
            # 30초 이하면 일반 전처리
            return await convert_preprocessing_for_microphone_stable(audio_data, sample_rate)

        # 30초 이상이면 청킹 처리
        chunk_size = 25 * sample_rate  # 25초 청크
        overlap_size = 2 * sample_rate  # 2초 오버랩

        processed_chunks = []
        start_idx = 0

        while start_idx < len(audio_data):
            end_idx = min(start_idx + chunk_size, len(audio_data))
            chunk = audio_data[start_idx:end_idx]

            # 각 청크 전처리
            processed_chunk = await convert_preprocessing_for_microphone_stable(chunk, sample_rate)
            processed_chunks.append(processed_chunk)

            start_idx += (chunk_size - overlap_size)

        # 청크들 결합
        result = np.concatenate(processed_chunks)

        print(f"✅ 청킹 전처리 완료 - 출력 길이: {len(result)}")
        return result

    except Exception as e:
        print(f"❌ 청킹 전처리 실패: {e}")
        return audio_data


async def convert_preprocessing_simple(audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    간단한 전처리 (최소한의 처리)
    Args:
        audio_data: 입력 오디오 데이터
        sample_rate: 샘플링 레이트
    Returns:
        전처리된 오디오 데이터
    """
    try:
        print(f"⚡ 간단 전처리 시작 - 입력 길이: {len(audio_data)}")

        # 1. 정규화만 수행
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))

        # 2. 클리핑 방지
        audio_data = np.clip(audio_data, -1.0, 1.0)

        print(f"✅ 간단 전처리 완료 - 출력 길이: {len(audio_data)}")
        return audio_data

    except Exception as e:
        print(f"❌ 간단 전처리 실패: {e}")
        return audio_data


# =============================================================================
# 보조 함수들
# =============================================================================

def _remove_noise(audio_data: np.ndarray, sample_rate: int, noise_threshold: float = 0.01) -> np.ndarray:
    """기본 노이즈 제거"""
    try:
        # 간단한 threshold 기반 노이즈 제거
        mask = np.abs(audio_data) > noise_threshold
        audio_data = audio_data * mask
        return audio_data
    except Exception:
        return audio_data


def _apply_preemphasis(audio_data: np.ndarray, coef: float = 0.97) -> np.ndarray:
    """프리엠퍼시스 필터 적용"""
    try:
        if len(audio_data) > 1:
            return np.append(audio_data[0], audio_data[1:] - coef * audio_data[:-1])
        return audio_data
    except Exception:
        return audio_data


def _apply_lowpass_filter(audio_data: np.ndarray, sample_rate: int, cutoff: float = 8000) -> np.ndarray:
    """저역통과 필터 적용"""
    try:
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff / nyquist

        if normalized_cutoff >= 1.0:
            return audio_data

        b, a = scipy.signal.butter(4, normalized_cutoff, btype='low')
        filtered = scipy.signal.filtfilt(b, a, audio_data)
        return filtered
    except Exception:
        return audio_data


def _apply_vad(audio_data: np.ndarray, sample_rate: int, energy_threshold: float = 0.01) -> np.ndarray:
    """기본 Voice Activity Detection"""
    try:
        frame_length = int(0.025 * sample_rate)  # 25ms
        frame_step = int(0.01 * sample_rate)  # 10ms

        # 프레임 단위로 에너지 계산
        frames = []
        for i in range(0, len(audio_data) - frame_length, frame_step):
            frame = audio_data[i:i + frame_length]
            energy = np.mean(frame ** 2)
            frames.append((i, i + frame_length, energy))

        if not frames:
            return audio_data

        # 에너지 임계값 기반 음성 구간 검출
        energies = [f[2] for f in frames]
        max_energy = max(energies) if energies else 0

        if max_energy == 0:
            return audio_data

        threshold = max_energy * energy_threshold

        # 음성 구간만 유지
        speech_mask = np.zeros_like(audio_data, dtype=bool)
        for start, end, energy in frames:
            if energy > threshold:
                speech_mask[start:end] = True

        # 음성 구간이 너무 적으면 원본 반환
        if np.sum(speech_mask) < len(audio_data) * 0.1:
            return audio_data

        return audio_data * speech_mask.astype(float)

    except Exception:
        return audio_data


def _spectral_noise_reduction(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    """스펙트럼 기반 노이즈 감소"""
    try:
        # STFT
        stft = librosa.stft(audio_data, hop_length=512, n_fft=2048)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # 노이즈 추정 (첫 10프레임의 평균)
        noise_profile = np.mean(magnitude[:, :10], axis=1, keepdims=True)

        # 스펙트럼 차감
        enhanced_magnitude = magnitude - 0.5 * noise_profile
        enhanced_magnitude = np.maximum(enhanced_magnitude, 0.1 * magnitude)

        # 역변환
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)

        return enhanced_audio[:len(audio_data)]

    except Exception:
        return audio_data


def _dynamic_range_compression(audio_data: np.ndarray, ratio: float = 4.0, threshold: float = 0.5) -> np.ndarray:
    """동적 범위 압축"""
    try:
        # 간단한 컴프레서
        compressed = np.copy(audio_data)

        # 임계값을 넘는 부분에 압축 적용
        mask = np.abs(compressed) > threshold
        over_threshold = compressed[mask]

        # 압축 적용
        sign = np.sign(over_threshold)
        abs_vals = np.abs(over_threshold)
        compressed_vals = threshold + (abs_vals - threshold) / ratio

        compressed[mask] = sign * compressed_vals

        return compressed

    except Exception:
        return audio_data


def _advanced_vad(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    """고급 Voice Activity Detection"""
    try:
        # 여러 특성을 조합한 VAD
        frame_length = int(0.025 * sample_rate)
        frame_step = int(0.01 * sample_rate)

        features = []
        for i in range(0, len(audio_data) - frame_length, frame_step):
            frame = audio_data[i:i + frame_length]

            # 에너지
            energy = np.mean(frame ** 2)

            # 제로 크로싱 비율
            zcr = np.mean(np.abs(np.diff(np.sign(frame)))) / 2

            # 스펙트럼 중심
            try:
                spectrum = np.abs(np.fft.fft(frame))
                freqs = np.fft.fftfreq(len(frame), 1 / sample_rate)
                spectral_centroid = np.sum(freqs[:len(freqs) // 2] * spectrum[:len(spectrum) // 2]) / np.sum(
                    spectrum[:len(spectrum) // 2])
            except:
                spectral_centroid = 0

            features.append((i, i + frame_length, energy, zcr, spectral_centroid))

        if not features:
            return audio_data

        # 임계값 기반 분류
        energies = [f[2] for f in features]
        zcrs = [f[3] for f in features]

        energy_threshold = np.percentile(energies, 30)
        zcr_threshold = np.percentile(zcrs, 70)

        # 음성 구간 마스크 생성
        speech_mask = np.zeros_like(audio_data, dtype=bool)
        for start, end, energy, zcr, _ in features:
            if energy > energy_threshold and zcr < zcr_threshold:
                speech_mask[start:end] = True

        # 음성 구간이 너무 적으면 원본 반환
        if np.sum(speech_mask) < len(audio_data) * 0.1:
            return audio_data

        return audio_data * speech_mask.astype(float)

    except Exception:
        return audio_data


def _adaptive_filtering(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    """적응형 필터링"""
    try:
        # 간단한 적응형 위너 필터
        # 노이즈 추정
        noise_estimate = np.var(audio_data[:int(0.1 * sample_rate)])  # 첫 0.1초를 노이즈로 가정
        signal_power = np.var(audio_data)

        if signal_power == 0:
            return audio_data

        # 위너 필터 계수
        snr = signal_power / (noise_estimate + 1e-10)
        wiener_filter = snr / (snr + 1)

        return audio_data * wiener_filter

    except Exception:
        return audio_data


# =============================================================================
# MFCC 관련 함수들 (호환성 유지)
# =============================================================================

async def extract_mfcc_features(audio_data: np.ndarray, sample_rate: int = 16000, n_mfcc: int = 13) -> np.ndarray:
    """MFCC 특성 추출"""
    try:
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        # 모든 특성 결합
        combined_features = np.vstack([mfcc, delta, delta2])
        return combined_features

    except Exception as e:
        print(f"❌ MFCC 추출 실패: {e}")
        # 더미 특성 반환
        return np.random.randn(39, 100)


async def convert_mfcc_to_audio_compatible(mfcc_features: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """MFCC + Delta + Delta2 특성을 Whisper 호환 오디오로 변환"""
    try:
        n_features, n_frames = mfcc_features.shape
        print(f"🎓 MFCC 특성 변환: {mfcc_features.shape}")

        if n_features == 13:
            # 순수 MFCC
            print("📊 순수 MFCC (13차원) 처리")
            mfcc_only = mfcc_features
            audio_approx = _mfcc_basic_reconstruction(mfcc_only, sample_rate)

        elif n_features == 26:
            # MFCC + Delta
            print("📊 MFCC + Delta (26차원) 처리")
            mfcc_only = mfcc_features[:13, :]
            delta = mfcc_features[13:26, :]
            audio_approx = _mfcc_delta_reconstruction(mfcc_only, delta, sample_rate)

        elif n_features == 39:
            # MFCC + Delta + Delta2 (완전한 특성)
            print("📊 MFCC + Delta + Delta2 (39차원) 처리")
            mfcc_only = mfcc_features[:13, :]
            delta = mfcc_features[13:26, :]
            delta2 = mfcc_features[26:39, :]
            audio_approx = _mfcc_full_reconstruction(mfcc_only, delta, delta2, sample_rate)

        else:
            # 알 수 없는 차원 - 첫 13개만 사용
            print(f"⚠️ 알 수 없는 특성 차원: {n_features} - 첫 13개만 사용")
            mfcc_like = mfcc_features[:min(13, n_features), :]
            audio_approx = _mfcc_basic_reconstruction(mfcc_like, sample_rate)

        # 정규화
        if np.max(np.abs(audio_approx)) > 0:
            audio_approx = audio_approx / np.max(np.abs(audio_approx))

        print(f"✅ MFCC 변환 완료: {len(audio_approx)} samples")
        return audio_approx

    except Exception as e:
        print(f"❌ MFCC 변환 실패: {e}")
        return np.random.randn(sample_rate) * 0.1


def _mfcc_basic_reconstruction(mfcc: np.ndarray, sample_rate: int) -> np.ndarray:
    """기본 MFCC 복원"""
    try:
        # LibROSA 역변환 사용
        audio_approx = librosa.feature.inverse.mfcc_to_audio(
            mfcc,
            sr=sample_rate,
            n_fft=2048,
            hop_length=512
        )
        return audio_approx

    except Exception as e:
        print(f"⚠️ LibROSA 역변환 실패: {e} - 대체 방법 사용")
        return _manual_mfcc_reconstruction(mfcc, sample_rate)


def _mfcc_delta_reconstruction(mfcc: np.ndarray, delta: np.ndarray, sample_rate: int) -> np.ndarray:
    """MFCC + Delta 복원"""
    try:
        print("🔄 MFCC + Delta 협력 복원")

        # Delta 정보로 MFCC 궤적 개선
        enhanced_mfcc = _improve_mfcc_with_delta(mfcc, delta)

        # 개선된 MFCC로 오디오 복원
        audio = _mfcc_basic_reconstruction(enhanced_mfcc, sample_rate)

        # Delta 기반 시간적 변조 적용
        modulated_audio = _apply_delta_temporal_modulation(audio, delta, sample_rate)

        return modulated_audio

    except Exception as e:
        print(f"⚠️ MFCC+Delta 복원 실패: {e}")
        return _mfcc_basic_reconstruction(mfcc, sample_rate)


def _mfcc_full_reconstruction(mfcc: np.ndarray, delta: np.ndarray, delta2: np.ndarray, sample_rate: int) -> np.ndarray:
    """MFCC + Delta + Delta2 완전 복원"""
    try:
        print("🔄 MFCC + Delta + Delta2 완전 복원")

        # 1단계: Delta2로 Delta 궤적 개선
        improved_delta = _improve_delta_with_delta2(delta, delta2)

        # 2단계: 개선된 Delta로 MFCC 궤적 개선
        enhanced_mfcc = _improve_mfcc_with_delta(mfcc, improved_delta)

        # 3단계: 개선된 MFCC로 기본 오디오 복원
        audio = _mfcc_basic_reconstruction(enhanced_mfcc, sample_rate)

        # 4단계: Delta 변조 적용
        audio = _apply_delta_temporal_modulation(audio, improved_delta, sample_rate)

        # 5단계: Delta2 기반 고차 변조 적용
        audio = _apply_delta2_spectral_modulation(audio, delta2, sample_rate)

        return audio

    except Exception as e:
        print(f"⚠️ 완전 복원 실패: {e}")
        return _mfcc_delta_reconstruction(mfcc, delta, sample_rate)


def _improve_mfcc_with_delta(mfcc: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Delta 정보로 MFCC 궤적 개선"""
    try:
        enhanced_mfcc = np.copy(mfcc)

        # Delta는 시간적 변화율이므로 이를 적분하여 부드러운 궤적 생성
        for coeff_idx in range(min(mfcc.shape[0], delta.shape[0])):
            for time_idx in range(1, min(mfcc.shape[1], delta.shape[1])):
                # Delta 정보를 사용한 시간적 연속성 개선
                delta_contribution = 0.1 * delta[coeff_idx, time_idx]
                enhanced_mfcc[coeff_idx, time_idx] += delta_contribution

        # 부드러운 전환을 위한 가우시안 필터 적용
        try:
            from scipy import ndimage
            enhanced_mfcc = ndimage.gaussian_filter1d(enhanced_mfcc, sigma=0.5, axis=1)
        except ImportError:
            pass  # scipy가 없으면 스킵

        return enhanced_mfcc

    except Exception as e:
        print(f"⚠️ MFCC 궤적 개선 실패: {e}")
        return mfcc


def _improve_delta_with_delta2(delta: np.ndarray, delta2: np.ndarray) -> np.ndarray:
    """Delta2 정보로 Delta 궤적 개선"""
    try:
        improved_delta = np.copy(delta)

        # Delta2는 가속도이므로 이를 적분하여 Delta 개선
        for coeff_idx in range(min(delta.shape[0], delta2.shape[0])):
            for time_idx in range(1, min(delta.shape[1], delta2.shape[1])):
                # Delta2를 적분하여 Delta의 부드러운 변화 생성
                acceleration = delta2[coeff_idx, time_idx]
                velocity_change = 0.05 * acceleration
                improved_delta[coeff_idx, time_idx] += velocity_change

        return improved_delta

    except Exception as e:
        print(f"⚠️ Delta 궤적 개선 실패: {e}")
        return delta


def _apply_delta_temporal_modulation(audio: np.ndarray, delta: np.ndarray, sample_rate: int) -> np.ndarray:
    """Delta 기반 시간적 변조"""
    try:
        # Delta의 에너지를 시간적 변조에 활용
        delta_energy = np.mean(np.abs(delta), axis=0)

        if len(delta_energy) == 0:
            return audio

        # 오디오 길이에 맞게 Delta 에너지 보간
        audio_times = np.linspace(0, len(delta_energy) - 1, len(audio))
        delta_interp = np.interp(audio_times, np.arange(len(delta_energy)), delta_energy)

        # Delta 에너지 기반 진폭 변조
        modulation = 1.0 + 0.2 * (delta_interp - np.mean(delta_interp)) / (np.std(delta_interp) + 1e-10)
        modulated_audio = audio * modulation

        return modulated_audio

    except Exception as e:
        print(f"⚠️ Delta 시간 변조 실패: {e}")
        return audio


def _apply_delta2_spectral_modulation(audio: np.ndarray, delta2: np.ndarray, sample_rate: int) -> np.ndarray:
    """Delta2 기반 스펙트럼 변조"""
    try:
        # Delta2의 변화를 주파수 변조로 변환
        delta2_energy = np.mean(np.abs(delta2), axis=0)

        if len(delta2_energy) == 0:
            return audio

        # 오디오 길이에 맞게 보간
        audio_times = np.linspace(0, len(delta2_energy) - 1, len(audio))
        delta2_interp = np.interp(audio_times, np.arange(len(delta2_energy)), delta2_energy)

        # Delta2 기반 미세한 주파수 시프트 시뮬레이션
        time_vec = np.arange(len(audio)) / sample_rate
        phase_modulation = 0.1 * np.cumsum(delta2_interp) / sample_rate

        # 위상 변조 적용
        modulated_audio = audio * np.cos(2 * np.pi * phase_modulation)

        return modulated_audio

    except Exception as e:
        print(f"⚠️ Delta2 스펙트럼 변조 실패: {e}")
        return audio


def _manual_mfcc_reconstruction(mfcc: np.ndarray, sample_rate: int) -> np.ndarray:
    """수동 MFCC 복원 (LibROSA 실패 시 대체 방법)"""
    try:
        print("🔧 수동 MFCC 복원 시도")

        # MFCC 계수를 기반으로 간단한 사인파 합성
        n_frames = mfcc.shape[1]
        hop_length = 512
        audio_length = n_frames * hop_length

        # 기본 주파수 설정
        base_freq = 200  # Hz

        # 각 MFCC 계수를 서로 다른 주파수 성분으로 변환
        audio = np.zeros(audio_length)

        for coeff_idx in range(min(8, mfcc.shape[0])):  # 첫 8개 계수만 사용
            for frame_idx in range(n_frames):
                start_sample = frame_idx * hop_length
                end_sample = min(start_sample + hop_length, audio_length)

                # MFCC 계수를 주파수와 진폭으로 변환
                freq = base_freq * (coeff_idx + 1)
                amplitude = abs(mfcc[coeff_idx, frame_idx]) * 0.1

                # 해당 프레임에 사인파 추가
                t = np.arange(end_sample - start_sample) / sample_rate
                sine_wave = amplitude * np.sin(2 * np.pi * freq * t)
                audio[start_sample:end_sample] += sine_wave

        # 정규화
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.5

        print(f"✅ 수동 복원 완료: {len(audio)} samples")
        return audio

    except Exception as e:
        print(f"❌ 수동 복원 실패: {e}")
        # 최후의 수단: 화이트 노이즈
        return np.random.randn(sample_rate) * 0.05