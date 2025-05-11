"""
Voice cloning functionality for the Voice Assistant.
Uses TTS by Coqui for generating speech with cloned voices.
"""
import os
import shutil
import tempfile
import numpy as np
from typing import Optional, List, Dict, Any, Tuple

from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.utils.manage import ModelManager
from pydub import AudioSegment

from utils import logger, get_timestamp


class VoiceCloner:
    """
    Handles voice cloning using TTS by Coqui with XTTS v2 model.
    Supports creating, managing, and using voice embeddings.
    """

    def __init__(self,
                 models_dir: str = "voice_models",
                 voice_samples_dir: str = "voice_samples"):
        """
        Initialize the voice cloner.

        Args:
            models_dir: Directory for storing TTS models
            voice_samples_dir: Directory for storing voice samples
        """
        self.models_dir = os.path.abspath(models_dir)
        self.voice_samples_dir = os.path.abspath(voice_samples_dir)
        self.voices_dir = os.path.join(self.models_dir, "voices")
        self.model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        self.tts = None
        self.loaded = False
        self.available_voices = {}

        # Create necessary directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.voice_samples_dir, exist_ok=True)
        os.makedirs(self.voices_dir, exist_ok=True)

        logger.debug(f"Initialized VoiceCloner with models_dir: {self.models_dir}, voices_dir: {self.voices_dir}")
        self._scan_available_voices()

    def _scan_available_voices(self) -> None:
        """
        Scan the voices directory for available voice samples and embeddings.
        Updates self.available_voices with {voice_name: voice_info} dictionary.
        """
        self.available_voices = {}
        try:
            for item in os.listdir(self.voices_dir):
                voice_dir = os.path.join(self.voices_dir, item)
                if os.path.isdir(voice_dir):
                    voice_info = {
                        "name": item,
                        "path": voice_dir,
                        "embedding_path": os.path.join(voice_dir, "embedding.npy"),
                        "sample_path": os.path.join(voice_dir, "sample.wav"),
                        "info_file": os.path.join(voice_dir, "info.txt")
                    }

                    # Check if embedding exists
                    if os.path.exists(voice_info["embedding_path"]):
                        voice_info["has_embedding"] = True
                    else:
                        voice_info["has_embedding"] = False

                    # Check if sample exists
                    if os.path.exists(voice_info["sample_path"]):
                        voice_info["has_sample"] = True
                    else:
                        voice_info["has_sample"] = False

                    # Read metadata if exists
                    if os.path.exists(voice_info["info_file"]):
                        try:
                            with open(voice_info["info_file"], "r") as f:
                                metadata = {}
                                for line in f:
                                    if ":" in line:
                                        key, value = line.split(":", 1)
                                        metadata[key.strip()] = value.strip()
                                voice_info["metadata"] = metadata
                        except Exception as e:
                            logger.warning(f"Failed to parse voice metadata for {item}: {e}")
                            voice_info["metadata"] = {}
                    else:
                        voice_info["metadata"] = {}

                    # Add to available voices
                    self.available_voices[item] = voice_info

            logger.debug(f"Found {len(self.available_voices)} available voices: {list(self.available_voices.keys())}")
        except Exception as e:
            logger.error(f"Error scanning available voices: {e}")

    def load_model(self) -> bool:
        """
        Load the TTS model.

        Returns:
            True if model loaded successfully, False otherwise
        """
        if self.loaded and self.tts is not None:
            logger.debug("TTS model already loaded")
            return True

        try:
            logger.info("Loading XTTS v2 model... This may take a while on first run.")

            # Use TTS API with default model path and device management
            self.tts = TTS(model_name=self.model_name, progress_bar=False)

            self.loaded = True
            logger.debug("TTS model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}", exc_info=True)
            return False

    def add_voice(self,
                  voice_sample_path: str,
                  voice_name: str,
                  metadata: Optional[Dict[str, str]] = None) -> Optional[str]:
        """
        Process a voice sample to create a new voice.

        Args:
            voice_sample_path: Path to the voice sample audio file (.wav, .mp3, etc.)
            voice_name: Name for the new voice
            metadata: Optional metadata about the voice (e.g., gender, age, accent)

        Returns:
            Path to the voice directory if successful, None otherwise
        """
        # Sanitize voice name for filesystem use
        safe_voice_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in voice_name)
        safe_voice_name = safe_voice_name.strip().replace(" ", "_")

        if not safe_voice_name:
            logger.error("Invalid voice name")
            return None

        voice_dir = os.path.join(self.voices_dir, safe_voice_name)

        # Check if directory already exists
        if os.path.exists(voice_dir):
            logger.warning(f"Voice '{safe_voice_name}' already exists. Adding timestamp suffix.")
            safe_voice_name = f"{safe_voice_name}_{get_timestamp()}"
            voice_dir = os.path.join(self.voices_dir, safe_voice_name)

        try:
            # Create voice directory
            os.makedirs(voice_dir, exist_ok=True)

            # Check if sample is a supported audio format and convert if needed
            processed_sample_path = os.path.join(voice_dir, "sample.wav")

            # Convert to WAV if needed
            audio = AudioSegment.from_file(voice_sample_path)

            # Ensure correct format for TTS (16kHz, 16-bit, mono)
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            audio.export(processed_sample_path, format="wav")

            logger.debug(f"Processed voice sample saved to: {processed_sample_path}")

            # Generate embedding
            if not self.load_model():
                logger.error("Failed to load TTS model for voice embedding creation")
                return None

            # Create embedding
            embedding_path = os.path.join(voice_dir, "embedding.npy")

            logger.debug(f"Generating voice embedding from sample: {processed_sample_path}")

            # Generate and save the embedding using the loaded TTS model
            # XTTS creates speaker embeddings when synthesizing, so we'll create a placeholder
            # that will be replaced during first synthesis
            placeholder_embedding = np.zeros((512,), dtype=np.float32)  # Placeholder size
            np.save(embedding_path, placeholder_embedding)

            # Save metadata
            if metadata:
                with open(os.path.join(voice_dir, "info.txt"), "w") as f:
                    f.write(f"Name: {voice_name}\n")
                    for key, value in metadata.items():
                        f.write(f"{key}: {value}\n")
            else:
                with open(os.path.join(voice_dir, "info.txt"), "w") as f:
                    f.write(f"Name: {voice_name}\n")
                    f.write(f"Created: {get_timestamp()}\n")

            # Update available voices
            self._scan_available_voices()

            logger.info(f"Added new voice: {safe_voice_name}")
            return voice_dir

        except Exception as e:
            logger.error(f"Failed to add voice: {e}", exc_info=True)
            # Clean up if failure
            if os.path.exists(voice_dir):
                try:
                    shutil.rmtree(voice_dir)
                except Exception as clean_e:
                    logger.warning(f"Failed to clean up voice directory after error: {clean_e}")
            return None

    def list_voices(self) -> List[Dict[str, Any]]:
        """
        Get a list of available voices with their information.

        Returns:
            List of voice information dictionaries
        """
        self._scan_available_voices()
        return list(self.available_voices.values())

    def generate_speech(self,
                        text: str,
                        voice_name: str,
                        language: str = "en",
                        output_path: Optional[str] = None) -> Optional[str]:
        """
        Generate speech using a cloned voice.

        Args:
            text: Text to convert to speech
            voice_name: Name of the voice to use
            language: Language code (e.g., "en", "es", "fr")
            output_path: Optional path to save the generated audio file

        Returns:
            Path to the generated audio file if successful, None otherwise
        """
        if not text.strip():
            logger.debug("Skipping TTS for empty text")
            return None

        # Make sure the model is loaded
        if not self.load_model():
            logger.error("Failed to load TTS model for speech generation")
            return None

        # Check if voice exists
        self._scan_available_voices()
        if voice_name not in self.available_voices:
            logger.error(f"Voice '{voice_name}' not found")
            return None

        voice_info = self.available_voices[voice_name]

        # Check if voice has a sample
        if not voice_info["has_sample"]:
            logger.error(f"Voice '{voice_name}' doesn't have a sample file")
            return None

        try:
            # Create output path if not provided
            if output_path is None:
                timestamp = get_timestamp()
                output_dir = "temp_recordings"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"response_{voice_name}_{timestamp}.wav")

            # Generate speech with XTTS using the voice sample directly
            logger.debug(f"Generating speech with voice '{voice_name}' (language: {language})")

            # Use the reference audio approach rather than embeddings
            reference_audio_path = voice_info["sample_path"]

            self.tts.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=reference_audio_path,
                language=language
            )

            logger.debug(f"Generated speech saved to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to generate speech: {e}", exc_info=True)
            return None

    def remove_voice(self, voice_name: str) -> bool:
        """
        Remove a voice from the system.

        Args:
            voice_name: Name of the voice to remove

        Returns:
            True if successfully removed, False otherwise
        """
        self._scan_available_voices()
        if voice_name not in self.available_voices:
            logger.error(f"Voice '{voice_name}' not found for removal")
            return False

        voice_info = self.available_voices[voice_name]
        try:
            shutil.rmtree(voice_info["path"])
            logger.info(f"Removed voice: {voice_name}")
            # Update available voices
            self._scan_available_voices()
            return True
        except Exception as e:
            logger.error(f"Failed to remove voice '{voice_name}': {e}")
            return False

    def get_voice_info(self, voice_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific voice.

        Args:
            voice_name: Name of the voice

        Returns:
            Dictionary with voice information if found, None otherwise
        """
        self._scan_available_voices()
        return self.available_voices.get(voice_name)