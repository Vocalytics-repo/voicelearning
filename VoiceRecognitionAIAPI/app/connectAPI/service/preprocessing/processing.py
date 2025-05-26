import numpy as np
import librosa
# PCM 파일 로드 함수
async def convert_preprocessing(np_pcm, sr=16000, bit_depth=16)->np.ndarray:
    emphasized = apply_preemphasis(np_pcm)
    denoised = simple_noise_reduction(emphasized)
    normalized = normalize_audio(denoised)
    vad_mask, vad_frames = simple_vad(normalized)
    speech_only = normalized * vad_mask
    mfcc_features = compute_mfcc(speech_only)
    delta_features, delta2_features = compute_deltas(mfcc_features)
    combined_features = np.vstack([mfcc_features, delta_features, delta2_features])
    return combined_features


# 1. 프리엠퍼시스 (Pre-emphasis)
def apply_preemphasis(audio_data, coef=0.97):
    """Apply pre-emphasis filter to audio data"""
    return np.append(audio_data[0], audio_data[1:] - coef * audio_data[:-1])


# 2. 프레임 분할 및 윈도잉 (Framing & Windowing)
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


# 3. 스펙트럼 분석 (Spectral Analysis)
def compute_power_spectrum(frames, nfft=512):
    """Compute the power spectrum of each frame using FFT"""
    mag_frames = np.absolute(np.fft.rfft(frames, nfft))
    pow_frames = (1.0 / nfft) * (mag_frames ** 2)
    return pow_frames


# 4. 멜 필터뱅크 (Mel Filter Bank) - librosa 활용
def get_mel_filterbanks(nfilt=40, nfft=512, sample_rate=16000, low_freq=0, high_freq=None):
    """Create a Mel filter bank using librosa"""
    if high_freq is None:
        high_freq = sample_rate / 2

    # librosa의 mel filterbank 생성 함수 사용
    mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=nfft, n_mels=nfilt,
                                    fmin=low_freq, fmax=high_freq)
    return mel_basis


# 5. MFCC (Mel-Frequency Cepstral Coefficients) - librosa 활용
def compute_mfcc(audio_data, sample_rate, num_cepstral=13):
    """Compute MFCC features from an audio signal using librosa"""
    # librosa의 MFCC 추출 함수 사용
    mfcc_features = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=num_cepstral)
    return mfcc_features


# 6. 델타 및 델타-델타 특성 (Delta and Delta-Delta Features) - librosa 활용
def compute_deltas(features):
    """Compute delta features using librosa"""
    delta_features = librosa.feature.delta(features)
    delta2_features = librosa.feature.delta(features, order=2)
    return delta_features, delta2_features


# 7. 노이즈 제거 (Noise Reduction) - 간단한 방법
def simple_noise_reduction(audio_data, sample_rate, noise_threshold=0.005):
    """Apply simple noise reduction by thresholding"""
    # 간단한 임계값 기반 노이즈 제거
    denoised = np.copy(audio_data)
    denoised[np.abs(denoised) < noise_threshold] = 0
    return denoised


# 8. 정규화 기법 (Normalization Techniques)
def normalize_audio(audio_data, method='peak'):
    """Normalize audio data"""
    if method == 'peak':
        # 피크 기반 정규화 (-1 ~ 1 범위)
        return audio_data / np.max(np.abs(audio_data))
    elif method == 'rms':
        # RMS 기반 정규화
        rms = np.sqrt(np.mean(audio_data ** 2))
        return audio_data / (rms * 10)  # -0.1 ~ 0.1 범위
    else:
        return audio_data


# 9. VAD (Voice Activity Detection) - 에너지 기반 간단 구현
def simple_vad(audio_data, sample_rate, frame_size=0.025, frame_stride=0.01, energy_threshold=0.1):
    """Simple energy-based Voice Activity Detection"""
    # 프레임 길이와 간격을 샘플 단위로 변환
    frame_length = int(frame_size * sample_rate)
    frame_step = int(frame_stride * sample_rate)

    # 프레임 분할
    frames = frame_signal(audio_data, sample_rate, frame_size, frame_stride)

    # 각 프레임의 에너지 계산
    energy = np.sum(frames ** 2, axis=1)

    # 에너지 정규화
    energy = energy / np.max(energy)

    # 임계값 적용
    speech_frames = energy > energy_threshold

    # 프레임 단위 결정을 샘플 단위로 변환
    # 정확한 길이의 마스크 생성
    speech_mask = np.zeros_like(audio_data)

    for i, is_speech in enumerate(speech_frames):
        start = i * frame_step
        end = min(start + frame_length, len(audio_data))
        speech_mask[start:end] = 1 if is_speech else 0

    # 정확히 원본 오디오 길이와 일치하도록 함
    speech_mask = speech_mask[:len(audio_data)]

    return speech_mask, speech_frames


# 10. 시간 스케일링 (Time Scaling) - librosa 활용
def simple_time_scale(audio_data, scale_factor=1.0):
    """Scale the speed of an audio signal without changing pitch using librosa"""
    try:
        return librosa.effects.time_stretch(audio_data, rate=scale_factor)
    except TypeError:
        # 이전 librosa 버전 호환성
        try:
            return librosa.effects.time_stretch(y=audio_data, rate=scale_factor)
        except:
            print("librosa.effects.time_stretch 함수에 문제가 있습니다. 오디오 원본을 반환합니다.")
            return audio_data