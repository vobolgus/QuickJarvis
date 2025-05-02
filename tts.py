"""
Text-to-speech functionality using system commands for the Voice Assistant.
"""
import os
import subprocess
import time
from typing import Optional, List

from utils import logger, get_timestamp

class SystemTTS:
    """
    Handles text-to-speech synthesis using system commands.
    Utilizes direct speech synthesis without saving to file when possible.
    """

    def __init__(self, language: str = "en", voice: Optional[str] = None):
        """
        Initialize the system TTS.

        Args:
            language: Language code (en, ru, etc.)
            voice: Optional voice name
        """
        self.language = language

        # Check system platform
        self.platform = self._get_platform()
        logger.info(f"Initializing System TTS on platform: {self.platform}")

        # Get available voices
        self.available_voices = self.get_available_voices()
        logger.info(f"Available voices: {self.available_voices}")

        # Set voice based on language and availability
        self.voice = voice or self._get_default_voice(language)
        logger.info(f"Using voice: {self.voice}")

    def _get_platform(self) -> str:
        """Determine the operating system platform."""
        import platform
        return platform.system().lower()

    def _get_default_voice(self, language: str) -> str:
        """Get default voice based on language and availability."""
        if not self.available_voices:
            return ""

        # Try to find language-appropriate voice
        if self.platform == "darwin":
            if language.lower() == "ru":
                # Возможные голоса для русского
                russian_voices = ["Yuri", "Milena", "Katya"]
                for voice in russian_voices:
                    if voice in self.available_voices:
                        return voice

                # Голоса, которые могут поддерживать кириллицу
                backup_voices = ["Tessa", "Moira", "Samantha"]
                for voice in backup_voices:
                    if voice in self.available_voices:
                        return voice
            else:
                # Английские голоса
                english_voices = ["Samantha", "Victoria", "Daniel", "Karen"]
                for voice in english_voices:
                    if voice in self.available_voices:
                        return voice

        # Если ничего не найдено, используем первый доступный голос
        return self.available_voices[0] if self.available_voices else ""

    def get_available_voices(self) -> List[str]:
        """Get list of available voices on the system."""
        voices = []

        if self.platform == "darwin":
            try:
                result = subprocess.run(['say', '-v', '?'],
                                       capture_output=True, text=True, check=True)

                for line in result.stdout.split('\n'):
                    if line.strip():
                        parts = line.split(' ', 1)
                        if parts:
                            name = parts[0].strip()
                            voices.append(name)

                logger.info(f"Found {len(voices)} voices on macOS")
                return voices
            except subprocess.SubprocessError as e:
                logger.error(f"Error getting voices: {str(e)}")
                return []
        else:
            logger.warning("Getting available voices not implemented for this platform")
            return []

    def generate_speech(self, text: str, **kwargs) -> Optional[str]:
        """
        Convert text to speech using direct system commands.

        Args:
            text: Text to convert to speech

        Returns:
            Path to a dummy audio file for compatibility
        """
        try:
            # Generate a unique filename for compatibility
            timestamp = get_timestamp()
            output_filename = f"response_{timestamp}.wav"

            logger.info(f"Generating direct speech with system command")
            start_time = time.time()

            # Use platform-specific commands for direct speech
            success = False

            if self.platform == "darwin":  # macOS
                # Get voice parameter if available
                voice_param = []
                if self.voice and self.voice in self.available_voices:
                    voice_param = ['-v', self.voice]

                # Direct speech with 'say' command
                cmd = ['say'] + voice_param + [text]
                logger.debug(f"Running direct speech command: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
                success = True

            elif self.platform == "linux":
                # Use espeak for Linux
                cmd = ['espeak', text]
                try:
                    subprocess.run(cmd, check=True)
                    success = True
                except (subprocess.SubprocessError, FileNotFoundError):
                    logger.error("Failed to use espeak. Please install it with: sudo apt-get install espeak")

            elif self.platform == "windows":
                # Use PowerShell for Windows
                ps_script = f"""
                Add-Type -AssemblyName System.Speech
                $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer
                $speak.Speak("{text.replace('"', '`"')}")
                """
                subprocess.run(['powershell', '-ExecutionPolicy', 'Bypass', '-Command', ps_script], check=True)
                success = True

            else:
                logger.error(f"Unsupported platform: {self.platform}")

            end_time = time.time()

            if success:
                logger.info(f"Speech synthesis completed in {end_time - start_time:.2f} seconds")

                # Create a dummy WAV file for compatibility with the rest of the system
                with open(output_filename, 'wb') as f:
                    # Write minimal WAV header (44 bytes, essentially an empty WAV file)
                    f.write(b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00')

                logger.info(f"Direct speech completed, dummy file created at {output_filename}")
                return output_filename
            else:
                logger.error("Speech synthesis failed")
                return None

        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}")

            # Try file-based speech as a fallback (only for macOS)
            if self.platform == "darwin":
                try:
                    logger.info("Attempting fallback to file-based speech...")

                    voice_param = []
                    if self.voice and self.voice in self.available_voices:
                        voice_param = ['-v', self.voice]

                    output_filename = f"response_{timestamp}_fallback.wav"
                    cmd = ['say'] + voice_param + ['-o', output_filename, '--file-format=WAVE', text]

                    subprocess.run(cmd, check=True)
                    logger.info(f"Fallback succeeded, file saved to {output_filename}")
                    return output_filename
                except Exception as e2:
                    logger.error(f"Fallback also failed: {str(e2)}")

            return None