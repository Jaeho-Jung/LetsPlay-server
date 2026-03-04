import os
import asyncio
from src import log
from faster_whisper import WhisperModel


class WhisperService:
    """
    Faster-Whisper (CTranslate2) 기반 음성 인식 서비스.

    HuggingFace Pipeline 대신 CTranslate2 엔진을 사용하여
    추론 속도 2~3배 향상 및 VRAM 사용량을 대폭 절감합니다.
    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        """Singleton pattern: creates only one instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_path: str = utils.MODEL_NAME,
        language: str = "ko",
        task: str = "transcribe",
        device: str = "auto",
        compute_type: str = "float16",
        beam_size: int = 5,
    ):
        """
        Initialize the WhisperService instance.

        Args:
            model_path: Path to the CTranslate2 converted model directory.
            language: Language code for speech recognition. Default is 'ko' (Korean).
            task: Task type. 'transcribe' or 'translate'.
            device: Device. 'auto', 'cuda', 'cpu' can be selected.
            compute_type: Computation precision. 'float16', 'int8_float16', 'int8', etc.
            beam_size: Beam search size. Larger means more accurate but slower.
        """
        if WhisperService._initialized:
            return

        self.model_path = model_path
        self.language = language
        self.task = task
        self.beam_size = beam_size

        # 디바이스 자동 감지
        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        self.compute_type = compute_type if self.device == "cuda" else "float32"

        try:
            log.info("Starting Faster-Whisper service...")
            self.model = self._load_model()
            WhisperService._initialized = True
            log.info("Faster-Whisper service started successfully!")
        except Exception as e:
            log.error(f"Error during Faster-Whisper service initialization: {e}")
            raise

    def _load_model(self) -> WhisperModel:
        """
        Load the Faster-Whisper model based on CTranslate2.

        Returns:
            WhisperModel: The loaded Faster-Whisper model.
        """
        try:
            model = WhisperModel(
                self.model_path,
                device=self.device,
                compute_type=self.compute_type,
            )
            log.info(
                f"Model loaded: device={self.device}, "
                f"compute_type={self.compute_type}"
            )
            return model
        except Exception as e:
            log.error(f"Error during model loading: {e}")
            raise

    def transcribe_sync(self, audio_path: str) -> str:
        """
        Synchronously transcribe an audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            The transcription of the audio file.
        """
        try:
            log.info(f"Transcribing audio file: {audio_path}")
            segments, info = self.model.transcribe(
                audio_path,
                language=self.language,
                task=self.task,
                beam_size=self.beam_size,
            )

            transcription = "".join(segment.text for segment in segments)

            log.info(
                f"Transcription completed: language={info.language}, "
                f"language_probability={info.language_probability:.2f}, "
                f"duration={info.duration:.1f}s"
            )
            return transcription
        except Exception as e:
            log.error(f"Error during transcription: {e}")
            raise

    async def transcribe_audio(self, audio_path: str) -> str:
        """
        Asynchronously transcribe an audio file using the Whisper model.

        Args:
            audio_path: Path to the audio file to be transcribed.

        Returns:
            The transcription of the audio file.
        """
        loop = asyncio.get_event_loop()
        transcript = await loop.run_in_executor(
            None,
            self.transcribe_sync,
            audio_path,
        )
        return transcript

    def transcribe_with_timestamps(self, audio_path: str) -> list[dict]:
        """
        Transcribe an audio file with timestamps.

        Args:
            audio_path: Path to the audio file.

        Returns:
            A list of dictionaries containing the start/end time and text of each segment.
        """
        try:
            log.info(f"Transcribing with timestamps: {audio_path}")
            segments, info = self.model.transcribe(
                audio_path,
                language=self.language,
                task=self.task,
                beam_size=self.beam_size,
            )

            results = [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                }
                for segment in segments
            ]

            log.info(f"Transcription completed: {len(results)} segments")
            return results
        except Exception as e:
            log.error(f"Error during transcription with timestamps: {e}")
            raise

    @classmethod
    def reset(cls):
        """Reset the singleton instance (mainly for testing)."""
        cls._instance = None
        cls._initialized = False