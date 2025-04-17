import os
import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration


# PCM 파일 로드 함수
def read_pcm(file_path, sr=16000, bit_depth=16):
    """
    PCM 파일을 읽어 numpy 배열로 반환합니다.
    """
    # PCM 파일 읽기 (16비트 또는 8비트 부호 있는 정수 가정)
    if bit_depth == 16:
        raw_data = np.fromfile(file_path, dtype=np.int16)
    elif bit_depth == 8:
        raw_data = np.fromfile(file_path, dtype=np.int8)
    else:
        raw_data = np.fromfile(file_path, dtype=np.int16)  # 기본값

    # [-1, 1] 범위로 정규화
    max_value = float(2 ** (bit_depth - 1))
    normalized_data = raw_data.astype(np.float32) / max_value

    return normalized_data, sr


# 모델 추론 함수
def transcribe_audio(model, processor, audio_file, device):
    """
    훈련된 모델을 사용하여 오디오 파일을 텍스트로 변환합니다.
    """
    # PCM 파일 로드
    audio_array, sr = read_pcm(audio_file)

    # 특성 추출
    input_features = processor.feature_extractor(
        audio_array,
        sampling_rate=sr,
        return_tensors="pt"
    ).input_features

    # 입력 특성을 모델과 동일한 장치로 이동
    input_features = input_features.to(device)

    # 모델을 통한 예측
    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    # 예측된 토큰을 텍스트로 디코딩
    transcription = processor.tokenizer.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    return transcription


def main():
    print("=== KsponSpeech_128001.pcm 파일 테스트 ===")

    # 테스트할 파일 경로
    test_file = "KsponSpeech_128001.pcm"  # 이미 찾은 경로 사용

    # 파일 존재 확인
    if not os.path.exists(test_file):
        print(f"오류: {test_file} 파일을 찾을 수 없습니다.")
        return

    print(f"테스트 파일: {test_file}")

    # 텍스트 파일 경로 (있는 경우)
    txt_file = test_file.replace('.pcm', '.txt')
    if os.path.exists(txt_file):
        with open(txt_file, 'r', encoding='utf-8') as f:
            reference_text = f.read().strip()
        print(f"참조 텍스트: {reference_text}")
    else:
        print("참조 텍스트 파일이 없습니다.")

    # 기기 설정 - CUDA 문제 해결을 위한 방법
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("CUDA 사용 가능 - GPU 사용")
    else:
        device = torch.device("cpu")
        print("CUDA 사용 불가 - CPU 사용")

    # 기본 모델 로드
    print("기본 Whisper 모델을 로드합니다...")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")

    # 모델을 장치로 명시적 이동
    model = model.to(device)
    print(f"모델을 {device}로 이동했습니다.")

    # 저장된 모델 파일을 찾아 로드 시도
    model_dir = "whisper_finetuned"

    # PT 파일 찾기
    pt_files = []

    # 현재 디렉토리에서 저장된 모델 파일 찾기
    if os.path.exists(model_dir):
        for root, _, files in os.walk(model_dir):
            for file in files:
                if file.endswith('.pt'):
                    pt_files.append(os.path.join(root, file))

    # 다른 일반적인 위치도 확인
    for location in [".", "./saved_model", "./models"]:
        if os.path.exists(location):
            for file in os.listdir(location):
                if file.endswith('.pt'):
                    pt_files.append(os.path.join(location, file))

    # pt 파일 찾았다면 로드 시도
    if pt_files:
        print(f"{len(pt_files)}개의 PT 파일을 찾았습니다:")
        for i, pt_file in enumerate(pt_files):
            print(f"  {i + 1}. {pt_file}")

        try:
            # 첫 번째 파일 로드 시도
            model_file = pt_files[0]
            print(f"모델 파일 로드 시도: {model_file}")

            # PT 파일을 장치에 맞게 로드
            state_dict = torch.load(model_file, map_location=device)
            model.load_state_dict(state_dict)
            print(f"모델 파일 로드 성공!")
        except Exception as e:
            print(f"모델 파일 로드 실패: {str(e)}")
            print("기본 Whisper 모델을 사용합니다.")
    else:
        print("저장된 PT 파일을 찾을 수 없습니다. 기본 Whisper 모델을 사용합니다.")

    # 모델을 평가 모드로 설정
    model.eval()

    # 음성 인식 수행
    print("\n음성 인식 중...")
    try:
        # 장치 정보를 transcribe_audio 함수에 전달
        transcription = transcribe_audio(model, processor, test_file, device)
        print("\n=== 인식 결과 ===")
        print(transcription)

        # 참조 텍스트가 있는 경우 WER 계산
        if os.path.exists(txt_file) and 'reference_text' in locals():
            try:
                import jiwer
                wer = jiwer.wer(reference_text, transcription)
                print(f"\nWER (Word Error Rate): {wer:.4f}")
            except ImportError:
                print("\nWER 계산을 위해 jiwer 라이브러리가 필요합니다.")
                print("pip install jiwer 명령으로 설치할 수 있습니다.")
    except Exception as e:
        print(f"음성 인식 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()  # 더 자세한 오류 정보 출력


if __name__ == "__main__":
    main()