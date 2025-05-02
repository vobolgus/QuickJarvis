"""
Text-to-speech functionality using Bark for the Voice Assistant.
"""
from typing import Optional

import torch
import numpy as np
from scipy.io.wavfile import write as write_wav
from transformers import BarkModel, BarkProcessor

from utils import logger, get_timestamp


class BarkTTS:
    """
    Handles text-to-speech synthesis using Suno's Bark model.
    """
    
    def __init__(self, 
                 model_id: str = "suno/bark", 
                 device: Optional[str] = None,
                 voice_preset: str = "v2/ru_speaker_0"):
        """
        Initialize the Bark text-to-speech system.
        
        Args:
            model_id: Identifier for the Bark model to load
            device: Device to use for model inference (None for auto-detection)
            voice_preset: Voice preset identifier to use for synthesis
        """
        self.model_id = model_id
        self.voice_preset = voice_preset
        
        # Set device
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Initialize model components
        logger.info(f"Loading Bark model {model_id}...")
        self._init_model()
    
    def _init_model(self) -> None:
        """Initialize the model and processor."""
        self.model = BarkModel.from_pretrained(
            self.model_id, 
            torch_dtype=self.torch_dtype
        )
        self.model.to(self.device)
        
        self.processor = BarkProcessor.from_pretrained(self.model_id)
        logger.info("Bark model loaded successfully")
    
    def generate_speech(self, text: str, voice_preset: Optional[str] = None) -> Optional[str]:
        """
        Convert text to speech using Bark and save to an audio file.
        
        Args:
            text: Text to convert to speech
            voice_preset: Optional override for the default voice preset
            
        Returns:
            Path to the generated audio file, or None if generation failed
        """
        try:
            preset = voice_preset if voice_preset else self.voice_preset
            logger.info(f"Generating speech with voice preset: {preset}")
            
            # Process text
            inputs = self.processor(
                text=text,
                voice_preset=preset,
                return_tensors="pt",
            ).to(self.device)
            
            # Generate speech
            logger.debug("Running Bark model inference...")
            speech_output = self.model.generate(**inputs, do_sample=True)
            audio_array = speech_output.cpu().numpy().squeeze()
            
            # Save audio to file
            timestamp = get_timestamp()
            output_filename = f"response_{timestamp}.wav"
            
            # Bark sample rate is 24000 Hz
            sample_rate = self.model.generation_config.sample_rate
            write_wav(output_filename, sample_rate, audio_array)
            
            logger.info(f"Speech generated and saved to {output_filename}")
            return output_filename
            
        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}")
            return None
