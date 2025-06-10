import os
import torch.nn as nn
import numpy as np
import torch
# import jiwer : label 텍스트가 있을 경우 활용
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperConfig
from typing import Any, Text


class sttmodel:
    def __init__(self):
        super(sttmodel, self).__init__()
        self.model = None
        self.processor = None
        self.device = None
        self.transcription = None
        self.sr = 16000
        self.is_finetuned_model = False

    def inference_stt(self, data):
        audio_array = data
        input_features = self.processor.feature_extractor(
            audio_array,
            sampling_rate=self.sr,
            return_tensors="pt"
        ).input_features
        # 입력 특성을 모델과 동일한 장치로 이동
        input_features = input_features.to(self.device)

        with torch.no_grad():
            if self.is_finetuned_model:
                # 파인튜닝된 모델 - 한국어 강제 지정
                predicted_ids = self.model.generate(
                    input_features,
                    max_length=200,
                    repetition_penalty=2.0,
                    no_repeat_ngram_size=3,
                    num_beams=1,
                    do_sample=False,
                    forced_decoder_ids=[[1, 50259], [2, 50264], [3, 50359]]  # 한국어 transcribe 강제
                )
            else:
                # 기본 Whisper 모델
                predicted_ids = self.model.generate(
                    input_features,
                    max_length=448,
                    num_beams=1,
                    do_sample=False
                )

        # 예측된 토큰을 텍스트로 디코딩
        transcription = self.processor.tokenizer.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        return transcription

    def initialize_model(self):
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

        # 기본 모델도 generation config 정리
        if hasattr(self.model, 'generation_config'):
            if hasattr(self.model.generation_config, 'forced_decoder_ids'):
                self.model.generation_config.forced_decoder_ids = None
            if hasattr(self.model.generation_config, 'suppress_tokens'):
                self.model.generation_config.suppress_tokens = None
            print("🔧 기본 모델 generation config 정리 완료")

        if self.device is not None:
            self.model.to(self.device)

    def check_cuda(self, ):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("GPU 사용")
        else:
            self.device = torch.device("cpu")
            print("CPU 사용")

    def _safe_load_checkpoint(self, model_file):
        """
        PyTorch 버전에 따라 안전하게 체크포인트를 로드
        """
        try:
            # PyTorch 2.6+ 대응: weights_only=False로 명시적 설정
            print("🔄 체크포인트 로딩 중...")
            checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)
            return checkpoint

        except TypeError:
            # 이전 PyTorch 버전에서는 weights_only 파라미터가 없음
            print("🔄 이전 PyTorch 버전으로 로딩 중...")
            checkpoint = torch.load(model_file, map_location=self.device)
            return checkpoint

        except Exception as e:
            print(f"⚠️ 첫 번째 로딩 시도 실패: {str(e)}")

            # 안전한 글로벌 설정으로 재시도
            try:
                from transformers.models.whisper.tokenization_whisper import WhisperTokenizer
                from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor

                # 안전한 글로벌로 등록
                safe_globals = [WhisperTokenizer, WhisperFeatureExtractor]

                # PyTorch 버전에 따른 처리
                if hasattr(torch.serialization, 'add_safe_globals'):
                    torch.serialization.add_safe_globals(safe_globals)
                    checkpoint = torch.load(model_file, map_location=self.device)
                else:
                    # 컨텍스트 매니저 사용
                    with torch.serialization.safe_globals(safe_globals):
                        checkpoint = torch.load(model_file, map_location=self.device)

                print("✅ 안전한 글로벌 설정으로 로딩 성공")
                return checkpoint

            except Exception as retry_error:
                print(f"❌ 재시도 실패: {str(retry_error)}")
                return None

    def create_compatible_processor(self, checkpoint):
        """
        버전 호환성을 고려한 프로세서 생성
        """
        try:
            stored_tokenizer = checkpoint['tokenizer']

            # 토크나이저 호환성 확인
            if not hasattr(stored_tokenizer, 'all_special_ids'):
                print("⚠️ 저장된 토크나이저 호환성 문제 - 새로 생성")
                raise AttributeError("Missing all_special_ids attribute")

            # FeatureExtractor 호환성 확인
            stored_feature_extractor = checkpoint['feature_extractor']
            required_attrs = ['dither', 'chunk_length', 'feature_size']
            missing_attrs = [attr for attr in required_attrs if not hasattr(stored_feature_extractor, attr)]

            if missing_attrs:
                print(f"⚠️ FeatureExtractor 호환성 문제 - 누락된 속성: {missing_attrs}")
                raise AttributeError(f"Missing attributes: {missing_attrs}")

            # 둘 다 호환된다면 사용
            processor = WhisperProcessor(feature_extractor=stored_feature_extractor, tokenizer=stored_tokenizer)
            print("✅ 저장된 프로세서 사용")
            return processor

        except (AttributeError, KeyError, Exception) as e:
            print(f"⚠️ 저장된 프로세서 사용 불가: {str(e)}")
            print("🔄 새로운 프로세서로 대체")

            # 완전히 새로운 프로세서 생성하되, 한국어로 강제 설정
            from transformers import WhisperProcessor
            new_processor = WhisperProcessor.from_pretrained("openai/whisper-small")

            # 한국어 토큰 ID 설정
            korean_token_id = 50264  # 한국어 토큰 ID
            transcribe_token_id = 50359  # transcribe 태스크 토큰 ID

            print("✅ 새로운 프로세서 생성 완료 (한국어 설정)")
            return new_processor

    def load_pt(self, pt_file: Any):
        try:
            model_file = pt_file
            print(f"모델 파일 로딩 시도: {model_file}")

            # 파일 존재 확인
            if not os.path.exists(model_file):
                print(f"❌ 파일이 존재하지 않습니다: {model_file}")
                return False

            # PyTorch 2.6+ 호환성을 위한 체크포인트 로드
            checkpoint = self._safe_load_checkpoint(model_file)
            if checkpoint is None:
                return False

            if 'model_state_dict' in checkpoint:
                # 파인튜닝된 모델 형식 (convert_to_pt.py로 생성된 파일)
                config = WhisperConfig.from_dict(checkpoint['model_config'])
                self.model = WhisperForConditionalGeneration(config)
                self.model.load_state_dict(checkpoint['model_state_dict'])

                # 호환성을 고려한 프로세서 생성
                self.processor = self.create_compatible_processor(checkpoint)

                # generation config 호환성 수정
                if hasattr(self.model, 'generation_config'):
                    # forced_decoder_ids 제거
                    if hasattr(self.model.generation_config, 'forced_decoder_ids'):
                        self.model.generation_config.forced_decoder_ids = None
                        print("🔧 forced_decoder_ids 제거됨")

                    # 기타 충돌 가능한 설정들도 정리
                    if hasattr(self.model.generation_config, 'suppress_tokens'):
                        self.model.generation_config.suppress_tokens = None
                        print("🔧 suppress_tokens 제거됨")

                    print("✅ Generation config 호환성 수정 완료")

                self.is_finetuned_model = True
                print("✅ 파인튜닝 모델 로드 완료")
            else:
                # 기존 방식 (단순 state_dict)
                state_dict = checkpoint
                self.model.load_state_dict(state_dict)
                self.is_finetuned_model = True
                print("✅ 모델 가중치 로드 완료")

            self.model.to(self.device)
            return True

        except Exception as e:
            print(f"❌ 모델 파일 에러 : {str(e)}")
            return False

    def get_model_path(self):
        """
        환경에 따라 모델 경로를 동적으로 결정
        """
        # 환경변수로 모델 경로 지정 (도커에서 유용)
        if 'MODEL_PATH' in os.environ:
            return os.environ['MODEL_PATH']

        # 현재 파일의 위치를 기준으로 상대 경로 계산
        current_file_dir = os.path.dirname(os.path.abspath(__file__))

        # 가능한 경로들을 우선순위대로 시도
        possible_paths = [
            # 현재 파일과 같은 디렉토리의 model 폴더
            os.path.join(current_file_dir, "model"),
            # 상위 디렉토리의 model 폴더
            os.path.join(os.path.dirname(current_file_dir), "model"),
            # service 폴더 내의 model 폴더
            os.path.join(current_file_dir, "service", "model"),
            # 절대 경로 (개발 환경용)
            "/Users/yongmin/Desktop/voicelearning/VoiceRecognitionAIAPI/app/connectAPI/service/model",
            # 도커 환경용 경로
            "/app/app/connectAPI/service/model"
        ]

        for path in possible_paths:
            if os.path.exists(path):
                print(f"📁 모델 디렉토리 발견: {path}")
                return path

        print("❌ 모델 디렉토리를 찾을 수 없습니다.")
        return None

    def find_pt(self, model_dir=None):
        """
        pt 파일을 찾아서 로드
        """
        if model_dir is None:
            model_dir = self.get_model_path()

        if model_dir is None:
            print("⚠️ 모델 디렉토리를 찾을 수 없어 기본 Whisper 모델을 사용합니다.")
            return False

        found = False
        print(f"🔍 모델 파일 탐색 중: {model_dir}")

        if os.path.exists(model_dir):
            # 특정 파일명으로 먼저 시도
            specific_file = os.path.join(model_dir, "whisper_pronunciation_model.pt")
            if os.path.exists(specific_file):
                print(f"🎯 특정 모델 파일 발견: {specific_file}")
                if self.load_pt(specific_file):
                    return True

            # 디렉토리 내 모든 .pt 파일 탐색
            for root, _, files in os.walk(model_dir):
                for file in files:
                    if file.endswith(".pt"):
                        full_path = os.path.join(root, file)
                        print(f"📄 모델 파일 발견: {full_path}")
                        if self.load_pt(full_path):
                            found = True
                            break
                if found:
                    break
        else:
            print(f"❌ 디렉토리가 존재하지 않습니다: {model_dir}")

        if not found:
            print("⚠️ pt 파일을 찾을 수 없어 기본 Whisper 모델을 사용합니다.")

        return found

    def stt_model(self, test_file):
        self.model.eval()

        # 현재 사용 중인 모델 타입 출력
        if self.is_finetuned_model:
            print("🎯 파인튜닝된 발음 모델 사용 중 (반복 방지 설정 적용)")
        else:
            print("🔸 기본 Whisper 모델 사용 중")

        self.transcription = self.inference_stt(test_file)

    def start_stt(self, test_file: np.ndarray) -> Any:
        print("🚀 STT 모델 시작")
        self.check_cuda()
        self.initialize_model()
        self.find_pt()  # 인자 없이 호출하면 자동으로 경로 탐색
        self.stt_model(test_file)
        return self.transcription