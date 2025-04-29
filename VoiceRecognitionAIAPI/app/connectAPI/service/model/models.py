import os

import numpy as np
import torch
#import jiwer : label 텍스트가 있을 경우 활용
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from typing import Any, Text

class sttmodel:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self.transcription = None
        self.sr = 16000

    def inference_stt(self, data):
        audio_array = data
        input_features = self.processor.feature_extractor(
            audio_array,
            sampling_rate = self.sr,
            return_tensors = "pt"
        ).input_features
        # 입력 특성을 모델과 동일한 장치로 이동
        input_features = input_features.to(self.device)

        # 모델을 통한 예측
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)

        # 예측된 토큰을 텍스트로 디코딩
        transcription = self.processor.tokenizer.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        return transcription

    def check_cuda(self,):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("GPU 사용")
        else:
            self.device = torch.device("cpu")
            print("CPU 사용")

    def load_pt(self,pt_file: Any):
        try:
            model_file = pt_file
            state_dict = torch.load(model_file, map_location=self.device)
            self.model.load_state_dict(state_dict)
        except Exception as e:
            print(f"모델 파일 에러 : {str(e)}")

    def find_pt(self,model_dir):
        if os.path.exists(model_dir):
            for root, _, files in os.walk(model_dir):
                for file in files:
                    if file.endswith(".pt"):
                        self.load_pt(file)

    def stt_model(self, test_file):
        self.model.eval()
        self.transcription = self.inference_stt(test_file)

    def start_stt(self, test_file: np.ndarray) -> Any:
        self.check_cuda()
        self.find_pt("model")   # pt 확장자로 된 pt파일 이름을 넣는 것이 아닌 pt 파일이 존재하는 디렉토리 이름을 넣는다.
        self.stt_model(test_file)
        return self.transcription
