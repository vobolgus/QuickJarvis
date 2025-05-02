"""
Speech-to-text functionality using Whisper for the Voice Assistant.
"""
import warnings
from typing import Dict, Any, Optional

import torch
from transformers import (
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor, 
    pipeline
)

from utils import logger

# Filter out the specific FutureWarning from Whisper
warnings.filterwarnings("ignore", message="The input name `inputs` is deprecated.*")


class WhisperTranscriber:
    """
    Handles speech-to-text transcription using OpenAI's Whisper model.
    """
    
    def __init__(self, model_id: str = "openai/whisper-large-v3-turbo", device: Optional[str] = None):
        """
        Initialize the Whisper transcription system.
        
        Args:
            model_id: Identifier for the Whisper model to load
            device: Device to use for model inference (None for auto-detection)
        """
        self.model_id = model_id
        
        # Set device
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Initialize model components
        logger.info(f"Loading Whisper model {model_id}...")
        self._init_model()
    
    def _init_model(self) -> None:
        """Initialize the model, processor, and pipeline."""
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        self.model.to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        
        # Create speech recognition pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            generate_kwargs={"max_new_tokens": 128},
            device=self.device
        )
        logger.info("Whisper model loaded successfully")
    
    def transcribe(self, audio_file: str) -> Dict[str, Any]:
        """
        Transcribe the given audio file to text.
        
        Args:
            audio_file: Path to the audio file to transcribe
            
        Returns:
            Dictionary containing the transcription result with at least a "text" key
        """
        logger.info(f"Transcribing audio file: {audio_file}")
        result = self.pipe(audio_file)
        logger.info("Transcription completed")
        return result
    
    def get_transcription_text(self, audio_file: str) -> str:
        """
        Get only the transcribed text from an audio file.
        
        Args:
            audio_file: Path to the audio file to transcribe
            
        Returns:
            Transcribed text as a string
        """
        result = self.transcribe(audio_file)
        return result["text"]
