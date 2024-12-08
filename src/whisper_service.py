import asyncio
import os
import torch
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor,
    logging as transformers_log,
)
from src import log
from src.utils import utils


class WhisperService:
    _initialized = False

    def __init__(self, language: str = 'ko'):
        """
        Initialize the WhisperService class. This ensures the Whisper model and related components are only initialized once.

        Args:
            language (str): Language code for the model. Default is 'ko' (Korean).
        """
        if not WhisperService._initialized:
            os.environ["TRANSFORMERS_VERBOSITY"] = "error"
            transformers_log.set_verbosity_error()
            self.model_name = utils.MODEL_NAME
            self.language = language
            self.task = utils.TASK
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            try:
                log.info("Starting Whisper service...")
                self.model = self._load_model()
                self.tokenizer = self._create_tokenizer()
                self.processor = self._create_processor()
                self.pipeline_asr = self._create_asr_pipeline()
                WhisperService._initialized = True
                log.info("Whisper service started successfully!")
            except Exception as e:
                log.error(f"Error during Whisper service initialization: {str(e)}")
                raise

    def _load_model(self) -> WhisperForConditionalGeneration:
        """
        Load the Whisper model for conditional generation.

        Returns:
            WhisperForConditionalGeneration: The loaded Whisper model.
        """
        try:
            model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                attn_implementation='sdpa'
            )
            model.to(self.device)
            log.info(f"Model loaded on {self.device.upper()}")
            return model
        except Exception as e:
            log.error(f"Error during Whisper model loading: {str(e)}")
            raise

    def _create_processor(self) -> WhisperProcessor:
        """
        Create the Whisper processor for audio processing.

        Returns:
            WhisperProcessor: The created Whisper processor.
        """
        try:
            processor = WhisperProcessor.from_pretrained(
                self.model_name,
                language=self.language,
                task=self.task
            )
            log.info("WhisperProcessor created successfully")
            return processor
        except Exception as e:
            log.error(f"Error during WhisperProcessor creation: {str(e)}")
            raise

    def _create_tokenizer(self) -> WhisperTokenizer:
        """
        Create the Whisper tokenizer for text tokenization.

        Returns:
            WhisperTokenizer: The created Whisper tokenizer.
        """
        try:
            tokenizer = WhisperTokenizer.from_pretrained(
                self.model_name,
                language=self.language,
                task=self.task
            )
            log.info("WhisperTokenizer created successfully")
            return tokenizer
        except Exception as e:
            log.error(f"Error during WhisperTokenizer creation: {str(e)}")
            raise

    def _create_asr_pipeline(self) -> AutomaticSpeechRecognitionPipeline:
        """
        Create the Whisper automatic speech recognition pipeline.

        Returns:
            AutomaticSpeechRecognitionPipeline: The created ASR pipeline.
        """
        try:
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=self.language, task=self.task)
            pipeline = AutomaticSpeechRecognitionPipeline(
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                generate_kwargs={"forced_decoder_ids": forced_decoder_ids, "max_new_tokens": 128},
                torch_dtype=self.torch_dtype,
                device=self.device
            )
            log.info("ASR pipeline created successfully")
            return pipeline
        except Exception as e:
            log.error(f"Error during ASR pipeline creation: {str(e)}")
            raise

    async def transcribe_audio(self, audio_path: str) -> str:
        """
        Asynchronously transcribe an audio file using the Whisper model.

        Args:
            audio_path (str): Path to the audio file to be transcribed.

        Returns:
            str: The transcription of the audio file.
        """
        try:
            loop = asyncio.get_event_loop()
            log.info(f"Transcribing audio file: {audio_path}")
            with torch.cuda.amp.autocast():
                transcript = await loop.run_in_executor(
                    None,
                    lambda: self.pipeline_asr(audio_path, batch_size=8)["text"]
                )
            log.info("Transcription completed successfully!")
            return transcript
        except Exception as e:
            log.error(f"Error during transcription: {str(e)}")
            raise
