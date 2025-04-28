import numpy as np

async def convert_pcm_to_numpy(pcm_data, sample_width) -> np.ndarray:
    # NumPy 배열 변환
    dtype_map = {
        1: np.uint8,  # 8-bit
        2: np.int16,  # 16-bit
        4: np.int32  # 32-bit
    }
    if sample_width not in dtype_map:
        raise ValueError(f"지원되지 않는 샘플 크기: {sample_width}바이트")

    np_pcm = np.frombuffer(pcm_data, dtype=dtype_map[sample_width])

    return np_pcm