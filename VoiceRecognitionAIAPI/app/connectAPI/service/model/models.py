import torch.nn as nn
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperConfig
from typing import Any, Text
import os
import time


class STTModelSingleton:
    """
    싱글톤 패턴으로 모델을 한 번만 로딩하는 STT 모델
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(STTModelSingleton, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            print("🎯 STT 모델 싱글톤 초기화 시작")
            self.model = None
            self.processor = None
            self.device = None
            self.sr = 16000
            self.is_finetuned_model = False

            # 모델 초기화 (한 번만 실행)
            self._setup_model()
            STTModelSingleton._initialized = True
            print("✅ STT 모델 싱글톤 초기화 완료")

    def _setup_model(self):
        """모델 설정 (한 번만 실행)"""
        print("🚀 STT 모델 설정 시작")
        self.check_cuda()
        self.initialize_model()
        self.find_pt()
        print("✅ STT 모델 설정 완료")

    def transcribe(self, audio):
        """통합된 음성 인식 함수 (MFCC 제거된 버전)"""
        print(f"🔍 추론 시작 - 입력 타입: {type(audio)}, 형태: {audio.shape}")

        try:
            # 🎯 MFCC 처리 완전 제거 - 원시 오디오만 사용
            if len(audio.shape) == 2:
                print("⚠️ 2D 배열 감지 - 원시 오디오로 변환")
                # 2D 배열이 들어오면 첫 번째 축만 사용 (MFCC 대신)
                audio = audio[0, :] if audio.shape[0] < audio.shape[1] else audio.flatten()
                print(f"🔄 2D → 1D 변환 완료 (길이: {len(audio)})")
            else:
                print("✅ 1D 배열 감지 - 원시 오디오 직접 사용")

            # 정규화
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
                print("✅ 오디오 정규화 완료")

            # 최소 길이 확보 (Whisper 요구사항)
            if len(audio) < 1600:
                repeat_times = (1600 // len(audio)) + 1
                audio = np.tile(audio, repeat_times)[:1600]
                print(f"🔄 최소 길이 확보: {len(audio)}")

            # Whisper 전처리
            input_features = self.processor.feature_extractor(
                audio,
                sampling_rate=self.sr,
                return_tensors="pt"
            ).input_features

            # GPU 사용 시 이동
            if torch.cuda.is_available():
                input_features = input_features.cuda()
                self.model = self.model.cuda()

            input_features = input_features.to(self.model.dtype)

            # 추론 수행
            start_time = time.time()

            with torch.no_grad():
                if self.is_finetuned_model:
                    # 🎯 파인튜닝 모델용 개선된 파라미터
                    predicted_ids = self.model.generate(
                        input_features,
                        max_length=448,
                        num_beams=3,  # 빔 서치로 정확도 향상
                        repetition_penalty=1.1,  # 낮게 조정
                        no_repeat_ngram_size=2,  # 낮게 조정
                        do_sample=False,
                        early_stopping=True,
                        forced_decoder_ids=[[1, 50259], [2, 50264], [3, 50359]]
                    )
                else:
                    # 🎯 원본 모델용 파라미터
                    predicted_ids = self.model.generate(
                        input_features,
                        max_length=448,
                        num_beams=5,  # 더 정확한 탐색
                        repetition_penalty=1.1,
                        no_repeat_ngram_size=2,
                        do_sample=False,
                        early_stopping=True,
                        forced_decoder_ids=[[1, 50259], [2, 50264], [3, 50359]]
                    )

            inference_time = time.time() - start_time
            print(f"🏁 전체 추론 시간: {inference_time:.3f}초")

            # 텍스트 디코딩
            transcription = self.processor.tokenizer.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]

            print(f"✅ 음성 인식 완료: {transcription}")
            return transcription

        except Exception as e:
            print(f"❌ 음성 인식 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            return "음성 인식 실패"

    # 🗑️ 기존 inference_stt 함수 제거 (transcribe로 통합)

    def initialize_model(self):
        try:
            print("🔄 기본 모델 및 프로세서 로딩 중...")
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
            self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

            if hasattr(self.model, 'generation_config'):
                if hasattr(self.model.generation_config, 'forced_decoder_ids'):
                    self.model.generation_config.forced_decoder_ids = None
                if hasattr(self.model.generation_config, 'suppress_tokens'):
                    self.model.generation_config.suppress_tokens = None
                print("🔧 기본 모델 generation config 정리 완료")

            if self.device is not None:
                self.model.to(self.device)

            print("✅ 기본 모델 초기화 완료")

        except Exception as e:
            print(f"❌ 기본 모델 초기화 실패: {str(e)}")
            # 최후의 수단: 수동으로 컴포넌트 생성
            try:
                from transformers import WhisperFeatureExtractor, WhisperTokenizer
                print("🔄 수동 컴포넌트 생성 시도...")

                feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
                tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small")
                self.processor = WhisperProcessor(
                    feature_extractor=feature_extractor,
                    tokenizer=tokenizer
                )
                self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

                if self.device is not None:
                    self.model.to(self.device)

                print("✅ 수동 모델 초기화 완료")

            except Exception as fallback_error:
                print(f"❌ 수동 초기화도 실패: {str(fallback_error)}")
                raise RuntimeError("모델 초기화가 완전히 실패했습니다.")

    def check_cuda(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("GPU 사용")
        else:
            self.device = torch.device("cpu")
            print("CPU 사용")

    def _safe_load_checkpoint(self, model_file):
        try:
            print("🔄 체크포인트 로딩 중...")
            checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)
            return checkpoint
        except TypeError:
            print("🔄 이전 PyTorch 버전으로 로딩 중...")
            checkpoint = torch.load(model_file, map_location=self.device)
            return checkpoint
        except Exception as e:
            print(f"⚠️ 첫 번째 로딩 시도 실패: {str(e)}")
            try:
                from transformers.models.whisper.tokenization_whisper import WhisperTokenizer
                from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor
                safe_globals = [WhisperTokenizer, WhisperFeatureExtractor]

                if hasattr(torch.serialization, 'add_safe_globals'):
                    torch.serialization.add_safe_globals(safe_globals)
                    checkpoint = torch.load(model_file, map_location=self.device)
                else:
                    with torch.serialization.safe_globals(safe_globals):
                        checkpoint = torch.load(model_file, map_location=self.device)

                print("✅ 안전한 글로벌 설정으로 로딩 성공")
                return checkpoint
            except Exception as retry_error:
                print(f"❌ 재시도 실패: {str(retry_error)}")
                return None

    def create_compatible_processor(self, checkpoint):
        try:
            stored_tokenizer = checkpoint['tokenizer']
            if not hasattr(stored_tokenizer, 'all_special_ids'):
                print("⚠️ 저장된 토크나이저 호환성 문제 - 새로 생성")
                raise AttributeError("Missing all_special_ids attribute")

            stored_feature_extractor = checkpoint['feature_extractor']
            required_attrs = ['dither', 'chunk_length', 'feature_size']
            missing_attrs = [attr for attr in required_attrs if not hasattr(stored_feature_extractor, attr)]

            if missing_attrs:
                print(f"⚠️ FeatureExtractor 호환성 문제 - 누락된 속성: {missing_attrs}")
                raise AttributeError(f"Missing attributes: {missing_attrs}")

            # WhisperProcessor 올바른 생성 방법
            processor = WhisperProcessor(
                feature_extractor=stored_feature_extractor,
                tokenizer=stored_tokenizer
            )
            print("✅ 저장된 프로세서 사용")
            return processor

        except (AttributeError, KeyError, Exception) as e:
            print(f"⚠️ 저장된 프로세서 사용 불가: {str(e)}")
            print("🔄 새로운 프로세서로 대체")

            # 안전한 새 프로세서 생성
            try:
                new_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
                print("✅ 새로운 프로세서 생성 완료 (한국어 설정)")
                return new_processor
            except Exception as processor_error:
                print(f"❌ 새 프로세서 생성 실패: {str(processor_error)}")
                # 최후의 수단: 기본 프로세서 생성
                from transformers import WhisperFeatureExtractor, WhisperTokenizer
                feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
                tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small")
                fallback_processor = WhisperProcessor(
                    feature_extractor=feature_extractor,
                    tokenizer=tokenizer
                )
                print("✅ Fallback 프로세서 생성 완료")
                return fallback_processor

    def load_pt(self, pt_file: Any):
        try:
            model_file = pt_file
            print(f"모델 파일 로딩 시도: {model_file}")

            if not os.path.exists(model_file):
                print(f"❌ 파일이 존재하지 않습니다: {model_file}")
                return False

            checkpoint = self._safe_load_checkpoint(model_file)
            if checkpoint is None:
                return False

            if 'model_state_dict' in checkpoint:
                config = WhisperConfig.from_dict(checkpoint['model_config'])
                self.model = WhisperForConditionalGeneration(config)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.processor = self.create_compatible_processor(checkpoint)

                if hasattr(self.model, 'generation_config'):
                    if hasattr(self.model.generation_config, 'forced_decoder_ids'):
                        self.model.generation_config.forced_decoder_ids = None
                        print("🔧 forced_decoder_ids 제거됨")
                    if hasattr(self.model.generation_config, 'suppress_tokens'):
                        self.model.generation_config.suppress_tokens = None
                        print("🔧 suppress_tokens 제거됨")
                    print("✅ Generation config 호환성 수정 완료")

                self.is_finetuned_model = True
                print("✅ 파인튜닝 모델 로드 완료")
            else:
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
        if 'MODEL_PATH' in os.environ:
            return os.environ['MODEL_PATH']

        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(current_file_dir, "model"),
            os.path.join(os.path.dirname(current_file_dir), "model"),
            os.path.join(current_file_dir, "service", "model"),
            "/Users/yongmin/Desktop/voicelearning/VoiceRecognitionAIAPI/app/connectAPI/service/model",
            "/app/app/connectAPI/service/model"
        ]

        for path in possible_paths:
            if os.path.exists(path):
                print(f"📁 모델 디렉토리 발견: {path}")
                return path

        print("❌ 모델 디렉토리를 찾을 수 없습니다.")
        return None

    def find_pt(self, model_dir=None):
        """파인튜닝 모델 파일 탐지 및 로드"""
        if model_dir is None:
            model_dir = self.get_model_path()

        if model_dir is None:
            print("⚠️ 모델 디렉토리를 찾을 수 없어 기본 Whisper 모델을 사용합니다.")
            return False

        found = False
        print(f"🔍 모델 파일 탐색 중: {model_dir}")

        if os.path.exists(model_dir):
            # 특정 파일명 우선 검색
            specific_file = os.path.join(model_dir, "whisper_pronunciation_model.pt")
            if os.path.exists(specific_file):
                print(f"🎯 특정 모델 파일 발견: {specific_file}")
                if self.load_pt(specific_file):
                    return True

            # 일반 .pt 파일 검색
            for root, _, files in os.walk(model_dir):
                for file in files:
                    if file.endswith(".pt") or file.endswith(".pth"):
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
            print("⚠️ .pt 파일을 찾을 수 없어 기본 Whisper 모델을 사용합니다.")

        return found


# 전역 모델 인스턴스 (서버 시작 시 한 번만 생성)
stt_model_instance = None


def get_stt_model():
    """STT 모델 인스턴스를 가져오는 함수"""
    global stt_model_instance
    if stt_model_instance is None:
        stt_model_instance = STTModelSingleton()
    return stt_model_instance


def quick_transcribe(audio_data: np.ndarray) -> str:
    """빠른 음성 인식 함수 (외부에서 호출용)"""
    model = get_stt_model()
    return model.transcribe(audio_data)