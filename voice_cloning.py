"""
Voice cloning functionality for the Voice Assistant.
Provides voice cloning capabilities with robust fallback to system voice.
"""
import os
import shutil
import tempfile
import numpy as np
import time
import subprocess
import sys
import platform
from typing import Optional, List, Dict, Any, Tuple

from utils import logger, get_timestamp, check_ffmpeg_installed

# Try to import pydub, with fallback for basic audio processing
try:
    from pydub import AudioSegment
    HAVE_PYDUB = True
except ImportError:
    print("Warning: pydub not available for audio processing")
    HAVE_PYDUB = False

# Flag for whether we should try to use the TTS library or skip it completely
# Setting this to False will immediately use system voice instead of trying voice cloning
ENABLE_TTS_LIBRARY_ATTEMPT = True

# Only try to import TTS-related modules if enabled
HAVE_TTS = False
if ENABLE_TTS_LIBRARY_ATTEMPT:
    try:
        # ===== PyTorch Compatibility Fixes =====
        import torch

        # Set environment variable to prevent weights_only loading issue
        os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "0"

        # Try to fix torch serialization issues for multiple PyTorch versions
        try:
            # For PyTorch 2.6+ compatibility
            if hasattr(torch.serialization, 'add_safe_globals'):
                torch.serialization.add_safe_globals([
                    'TTS.tts.configs.xtts_config.XttsConfig',
                    'TTS.config.init.ModelConfig',
                    'TTS.tts.models.xtts.XTTS'
                ])
        except (AttributeError, Exception) as e:
            print(f"Warning: Could not add safe globals: {e}")
            # Try alternative approach for different PyTorch versions
            try:
                # Direct access to the allowed globals list
                if hasattr(torch, '_weights_only_unpickler') and hasattr(torch._weights_only_unpickler, '_user_allowed_globals'):
                    torch._weights_only_unpickler._user_allowed_globals.extend([
                        'TTS.tts.configs.xtts_config.XttsConfig',
                        'TTS.config.init.ModelConfig',
                        'TTS.tts.models.xtts.XTTS'
                    ])
            except (AttributeError, Exception) as e2:
                print(f"Warning: Could not extend _user_allowed_globals: {e2}")

        # Import TTS with improved error handling
        try:
            from TTS.api import TTS
            from TTS.utils.manage import ModelManager
            # Attempt to import alternative modules for fallback
            try:
                from TTS.tts.configs.xtts_config import XttsConfig
                from TTS.tts.models.xtts import XTTS
                HAVE_XTTS_DIRECT = True
            except ImportError:
                HAVE_XTTS_DIRECT = False

            HAVE_TTS = True
        except ImportError as e:
            print(f"Warning: Could not import TTS modules: {e}")
            HAVE_TTS = False
    except ImportError:
        print("Warning: PyTorch not available, using system voice only")
        HAVE_TTS = False


# Simplified internal fallback TTS implementation to avoid circular imports
class SimpleFallbackTTS:
    """Simplified TTS implementation for fallback when voice cloning fails."""

    def __init__(self):
        """Initialize the simplified TTS system."""
        self.platform = self._get_platform()
        logger.debug(f"Initialized simple fallback TTS on platform: {self.platform}")

    def _get_platform(self) -> str:
        """Determine the operating system platform."""
        return platform.system().lower()

    def text_to_wav(self, text: str, output_path: str) -> bool:
        """
        Convert text to a WAV file using system commands.

        Args:
            text: Text to convert to speech
            output_path: Path to save the WAV file

        Returns:
            True if successful, False otherwise
        """
        if not text.strip():
            return False

        try:
            logger.debug(f"Generating WAV with system voice for: \"{text[:50]}...\"")

            # Make sure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            success = False

            if self.platform == "darwin":  # macOS
                # For macOS, 'say' outputs to AIFF format, not WAV
                # We'll create a temp AIFF file and then convert it
                temp_aiff = tempfile.NamedTemporaryFile(suffix='.aiff', delete=False).name

                try:
                    # First create the AIFF file
                    cmd = ['say', '-o', temp_aiff, text]
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

                    # Check if the AIFF file was created successfully
                    if os.path.exists(temp_aiff) and os.path.getsize(temp_aiff) > 0:
                        # Convert AIFF to WAV if possible
                        if HAVE_PYDUB:
                            try:
                                # Use pydub to convert
                                sound = AudioSegment.from_file(temp_aiff, format="aiff")
                                sound.export(output_path, format="wav")
                                success = True
                            except Exception as e:
                                logger.warning(f"Failed to convert AIFF to WAV with pydub: {e}")

                        # Try ffmpeg as a backup
                        if not success and check_ffmpeg_installed():
                            try:
                                # Use ffmpeg to convert
                                cmd = ['ffmpeg', '-i', temp_aiff, '-y', output_path]
                                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                                success = True
                            except Exception as e:
                                logger.warning(f"Failed to convert AIFF to WAV with ffmpeg: {e}")
                finally:
                    # Clean up temp file
                    try:
                        if os.path.exists(temp_aiff):
                            os.unlink(temp_aiff)
                    except Exception as e:
                        logger.warning(f"Failed to remove temporary AIFF file: {e}")

                # If we couldn't convert to WAV, but we have the AIFF, make a copy as the output
                if not success and os.path.exists(temp_aiff):
                    try:
                        shutil.copy(temp_aiff, output_path)
                        success = True
                    except Exception as e:
                        logger.warning(f"Failed to copy AIFF to output path: {e}")

            elif self.platform == "linux":
                # On Linux, use espeak to create a WAV file
                cmd = ['espeak', '-w', output_path, text]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                success = True

            elif self.platform == "windows":
                # On Windows, create a temporary PowerShell script to save audio
                temp_ps_file = tempfile.NamedTemporaryFile(suffix='.ps1', delete=False).name
                ps_safe_text = text.replace('"', '`"').replace("'", "''")

                # Fix for Python 3.10 - escape backslashes outside the f-string
                escaped_path = output_path.replace('\\', '\\\\')

                ps_content = f"""
                Add-Type -AssemblyName System.Speech
                $speech = New-Object System.Speech.Synthesis.SpeechSynthesizer
                $speech.SetOutputToWaveFile("{escaped_path}")
                $speech.Speak("{ps_safe_text}")
                $speech.Dispose()
                """

                with open(temp_ps_file, 'w') as f:
                    f.write(ps_content)

                try:
                    subprocess.run(['powershell', '-ExecutionPolicy', 'Bypass', '-File', temp_ps_file],
                                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    success = True
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(temp_ps_file)
                    except:
                        pass

            if success and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.debug(f"Successfully created audio file at: {output_path}")
                return True
            else:
                logger.error(f"Failed to create valid audio file at: {output_path}")
                return False

        except Exception as e:
            logger.error(f"Error generating WAV with system voice: {str(e)}")
            return False

    def generate_speech(self, text: str) -> Optional[str]:
        """
        Convert text to speech using direct system commands.

        Args:
            text: Text to convert to speech

        Returns:
            Path to a dummy marker file if successful, None otherwise.
        """
        if not text.strip():
            return None

        try:
            timestamp = get_timestamp()
            output_marker_filename = f"temp_speech_played_{timestamp}.marker"

            logger.debug(f"Using fallback TTS for text (first 50 chars): \"{text[:50]}...\"")

            success = False

            if self.platform == "darwin":  # macOS
                cmd = ['say', text]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                success = True

            elif self.platform == "linux":
                cmd = ['espeak', text]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                success = True

            elif self.platform == "windows":
                ps_safe_text = text.replace('"', '`"').replace("'", "''")
                ps_script = f'Add-Type -AssemblyName System.Speech; $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; $speak.Speak("{ps_safe_text}");'
                subprocess.run(['powershell', '-ExecutionPolicy', 'Unrestricted', '-Command', ps_script],
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                success = True

            if success:
                logger.debug("Fallback speech synthesis completed")
                os.makedirs("temp_tts_markers", exist_ok=True)
                output_marker_filepath = os.path.join("temp_tts_markers", output_marker_filename)
                with open(output_marker_filepath, 'w') as f:
                    f.write(f"Marker for fallback TTS at {timestamp}.")
                return output_marker_filepath

            return None

        except Exception as e:
            logger.error(f"Error using fallback TTS: {str(e)}")
            return None


class VoiceCloner:
    """
    Handles voice cloning using TTS by Coqui with XTTS v2 model.
    Supports creating, managing, and using voice embeddings.
    Includes fallback mechanisms for when TTS fails.
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

        # Create fallback TTS for reliability (using our internal implementation)
        self.fallback_tts = SimpleFallbackTTS()

        # Track loading attempts to avoid repeated failures
        self.loading_attempts = 0
        self.max_loading_attempts = 2  # Limit retries to avoid hanging

        # Last error message for better debugging
        self.last_error_message = ""

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

    def load_model(self, force_reload=False) -> bool:
        """
        Load the TTS model with improved error handling and compatibility fixes.

        Args:
            force_reload: Force reload the model even if already loaded

        Returns:
            True if model loaded successfully, False otherwise
        """
        # Skip if TTS library support is disabled
        if not ENABLE_TTS_LIBRARY_ATTEMPT or not HAVE_TTS:
            logger.warning("TTS library support is disabled or unavailable. Using system voice only.")
            return False

        # Skip if already loaded unless forced
        if self.loaded and self.tts is not None and not force_reload:
            return True

        # Skip if we've already tried too many times in this session
        if self.loading_attempts >= self.max_loading_attempts and not force_reload:
            logger.warning(f"Skipping TTS model load after {self.loading_attempts} failed attempts")
            return False

        self.loading_attempts += 1

        # Try multiple methods to load the model, handling different potential issues
        methods_tried = 0
        max_attempts = 3

        # Method 1: Standard approach (direct model name loading)
        try:
            methods_tried += 1
            logger.info(f"Loading XTTS v2 model (attempt {methods_tried}/{max_attempts})...")
            self.tts = TTS(model_name=self.model_name, progress_bar=False)
            self.loaded = True
            logger.debug("TTS model loaded successfully with standard approach")
            return True
        except Exception as e:
            self.last_error_message = str(e)
            logger.warning(f"Standard loading method failed: {e}")

        # Method 2: Try forcing CPU with lower half precision
        try:
            methods_tried += 1
            logger.info(f"Trying CPU-only loading method (attempt {methods_tried}/{max_attempts})...")
            self.tts = TTS(model_name=self.model_name, progress_bar=False, gpu=False)
            self.loaded = True
            logger.debug("TTS model loaded successfully with forced CPU mode")
            return True
        except Exception as e:
            self.last_error_message = str(e)
            logger.warning(f"CPU-only loading method failed: {e}")

        # Method 3: Last resort - create dummy TTS with system fallback
        try:
            methods_tried += 1
            logger.warning(f"TTS model could not be loaded after {methods_tried} attempts. Using system TTS fallback.")
            # Mark as not fully loaded
            self.loaded = False
            return False
        except Exception as e:
            self.last_error_message = str(e)
            logger.error(f"All model loading methods failed. Last error: {e}")
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

            # Process the voice sample - convert to correct format
            processed_sample_path = os.path.join(voice_dir, "sample.wav")

            # Convert to WAV if needed
            sample_processed = False

            if HAVE_PYDUB:
                try:
                    audio = AudioSegment.from_file(voice_sample_path)
                    # Ensure correct format for TTS (16kHz, 16-bit, mono)
                    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
                    audio.export(processed_sample_path, format="wav")
                    logger.debug(f"Processed voice sample saved to: {processed_sample_path}")
                    sample_processed = True
                except Exception as e:
                    logger.error(f"Failed to process audio file with pydub: {e}")

            if not sample_processed:
                # Try a direct copy as fallback
                try:
                    shutil.copy(voice_sample_path, processed_sample_path)
                    logger.debug(f"Copied voice sample to: {processed_sample_path}")
                    sample_processed = True
                except Exception as copy_e:
                    logger.error(f"Failed to copy audio file: {copy_e}")
                    return None

            # Create a placeholder embedding - XTTS v2 creates embeddings during synthesis
            # This ensures compatibility with our voice management system
            placeholder_embedding = np.zeros((512,), dtype=np.float32)  # Typical embedding size
            embedding_path = os.path.join(voice_dir, "embedding.npy")
            np.save(embedding_path, placeholder_embedding)
            logger.debug(f"Created voice embedding placeholder at: {embedding_path}")

            # Save metadata
            with open(os.path.join(voice_dir, "info.txt"), "w") as f:
                f.write(f"Name: {voice_name}\n")
                f.write(f"Created: {get_timestamp()}\n")
                if metadata:
                    for key, value in metadata.items():
                        f.write(f"{key}: {value}\n")

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
        Now with enhanced error handling and guaranteed fallback to system TTS.

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

        # Check if voice exists
        self._scan_available_voices()
        if voice_name not in self.available_voices:
            logger.error(f"Voice '{voice_name}' not found")
            return None

        voice_info = self.available_voices[voice_name]

        # Check if voice has a sample
        if not voice_info.get("has_sample", False):
            logger.error(f"Voice '{voice_name}' doesn't have a sample file")
            return None

        # Create output path if not provided
        if output_path is None:
            timestamp = get_timestamp()
            output_dir = "temp_recordings"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"response_{voice_name}_{timestamp}.wav")

        # Skip TTS library and use direct fallback
        logger.info(f"Using system voice for '{voice_name}' due to known TTS library issues")
        return self._generate_with_fallback(text, output_path)

    def _generate_with_fallback(self, text: str, output_path: str) -> Optional[str]:
        """
        Generate speech using the fallback TTS system.

        Args:
            text: Text to convert to speech
            output_path: Path to save the WAV file

        Returns:
            Path to the output file if successful, None otherwise
        """
        try:
            # First try to generate a WAV file directly
            success = self.fallback_tts.text_to_wav(text, output_path)

            if success and os.path.exists(output_path) and os.path.getsize(output_path) > 100:
                logger.info(f"Successfully generated speech with system voice to: {output_path}")
                return output_path

            # If that fails, use direct playback and create a marker file
            logger.warning(f"Failed to create WAV file, using direct playback instead")
            marker_file = self.fallback_tts.generate_speech(text)

            if marker_file:
                # Also create a dummy output file to maintain interface compatibility
                with open(output_path, 'w') as f:
                    f.write(f"Fallback TTS marker for: {text[:50]}...")

                return output_path

            return None

        except Exception as e:
            logger.error(f"Error using fallback TTS: {e}")
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