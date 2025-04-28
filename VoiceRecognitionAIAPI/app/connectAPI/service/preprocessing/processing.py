import numpy as np

# PCM 파일 로드 함수
def convert_preprocessing(np_pcm, sr=16000, bit_depth=16)->np.ndarray:

    # [-1, 1] 범위로 정규화
    max_value = float(2 ** (bit_depth - 1))
    normalized_data = np_pcm.astype(np.float32) / max_value

    return normalized_data