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
        self.platform = self._get_platform()
        logger.debug(f"Initializing System TTS on platform: {self.platform}")

        self.available_voices = self.get_available_voices()
        logger.debug(f"Available voices: {self.available_voices if self.available_voices else 'None found or not queried'}")

        self.voice = voice or self._get_default_voice(language)
        logger.debug(f"Using voice: {self.voice if self.voice else 'System default'}")

    def _get_platform(self) -> str:
        """Determine the operating system platform."""
        import platform
        return platform.system().lower()

    def _get_default_voice(self, language: str) -> str:
        """Get default voice based on language and availability."""
        if not self.available_voices: # if list is empty or None
            return ""

        # Try to find language-appropriate voice
        if self.platform == "darwin":
            lang_lower = language.lower()
            # Attempt to match language code like 'en_US' or 'ru_RU' first if present in voice details
            # For simplicity, we'll stick to the previous logic based on common voice names.
            if lang_lower == "ru" or lang_lower.startswith("ru-"):
                russian_voices = ["Yuri", "Milena", "Katya"]
                for v_name in russian_voices:
                    if v_name in self.available_voices: return v_name
                # Fallback for Russian on macOS if specific names not found
                for v_name in self.available_voices:
                    if "russian" in v_name.lower() or language.split('_')[0] in v_name.lower(): return v_name

            elif lang_lower == "en" or lang_lower.startswith("en-"):
                english_voices = ["Samantha", "Victoria", "Daniel", "Karen", "Alex", "Fred", "Tessa"]
                for v_name in english_voices:
                    if v_name in self.available_voices: return v_name
        # For other platforms or if no specific match, rely on system's default for the language,
        # or the first available voice as a last resort.
        return self.available_voices[0] if self.available_voices else ""


    def get_available_voices(self) -> List[str]:
        """Get list of available voices on the system."""
        voices = []
        if self.platform == "darwin":
            try:
                result = subprocess.run(['say', '-v', '?'],
                                       capture_output=True, text=True, check=True, encoding='utf-8')
                for line in result.stdout.split('\n'):
                    if line.strip():
                        parts = line.split('#', 1)[0].strip().split(None, 1) # Name is before # comment, split name from lang code
                        if parts and parts[0]:
                            voices.append(parts[0])
                logger.debug(f"Found {len(voices)} voices on macOS: {voices}")
                return voices
            except subprocess.SubprocessError as e:
                logger.error(f"Error getting voices on macOS: {str(e)}")
                return []
        # Add voice listing for other platforms if desired
        # elif self.platform == "linux":
        #   try: espeak --voices
        # elif self.platform == "windows":
        #   try: PowerShell Get-SpeechVoice
        else:
            logger.debug("Getting available voices not implemented for this platform.")
            return []

    def generate_speech(self, text: str, **kwargs) -> Optional[str]:
        """
        Convert text to speech using direct system commands.
        Creates a dummy file for compatibility if speech is played directly.

        Args:
            text: Text to convert to speech

        Returns:
            Path to a dummy audio file if successful, None otherwise.
        """
        try:
            timestamp = get_timestamp()
            # Dummy filename, as speech is played directly
            # A more robust system might not need this if playback isn't tied to files
            output_filename = f"temp_speech_played_{timestamp}.marker"

            logger.debug(f"Generating direct speech with system command for text: \"{text[:50]}...\"")
            start_time = time.time()
            success = False

            # Sanitize text slightly for command line (basic quote handling)
            # More robust sanitization might be needed depending on shell and content
            safe_text = text.replace('"', "'").replace("`", "'").replace("!", "").replace("$", "")


            if self.platform == "darwin":
                cmd = ['say']
                if self.voice and self.voice in self.available_voices:
                    cmd.extend(['-v', self.voice])
                cmd.append(safe_text) # 'say' handles text directly
                logger.debug(f"Running direct speech command: {' '.join(cmd)}")
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                success = True

            elif self.platform == "linux":
                cmd = ['espeak', f'"{safe_text}"'] # espeak prefers text quoted if it contains spaces
                logger.debug(f"Running direct speech command: {' '.join(cmd)}")
                try:
                    subprocess.run(cmd, check=True, shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) # shell=False with list of args
                    success = True
                except (subprocess.SubprocessError, FileNotFoundError) as e:
                    logger.error(f"Failed to use espeak: {e}. Please install it (e.g., sudo apt-get install espeak)")


            elif self.platform == "windows":
                # PowerShell needs careful quoting. Double quotes within the PowerShell string need to be escaped.
                ps_safe_text = safe_text.replace('"', '`"') # Escape double quotes for PowerShell
                ps_script = f"""
                Add-Type -AssemblyName System.Speech
                $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer
                """
                if self.voice: # Windows voice selection is more complex, typically by name
                     ps_script += f'$speak.SelectVoice("{self.voice}");\n' # This assumes self.voice is a valid Windows voice name
                ps_script += f'$speak.Speak("{ps_safe_text}");'

                logger.debug(f"Running PowerShell speech command with text: {ps_safe_text[:50]}...")
                subprocess.run(['powershell', '-ExecutionPolicy', 'Bypass', '-Command', ps_script], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                success = True
            else:
                logger.error(f"TTS not supported on platform: {self.platform}")

            end_time = time.time()

            if success:
                logger.debug(f"Direct speech synthesis completed in {end_time - start_time:.2f} seconds")
                # Create a dummy marker file for cleanup logic compatibility
                with open(output_filename, 'w') as f:
                    f.write("This is a marker for a directly played TTS audio.")
                return output_filename
            else:
                logger.error("Direct speech synthesis failed or not attempted.")
                return None

        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}", exc_info=True)
            # Fallback to file-based if direct fails (original code had this for macOS)
            if self.platform == "darwin" and isinstance(e, subprocess.CalledProcessError): # Only if command failed
                try:
                    logger.debug("Attempting fallback to file-based speech for macOS...")
                    output_filename_fallback = f"response_{timestamp}_fallback.wav" # Actual WAV
                    cmd_fallback = ['say']
                    if self.voice and self.voice in self.available_voices:
                        cmd_fallback.extend(['-v', self.voice])
                    cmd_fallback.extend(['-o', output_filename_fallback, '--file-format=WAVE', safe_text])
                    subprocess.run(cmd_fallback, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    logger.debug(f"Fallback to file succeeded, file saved to {output_filename_fallback}")
                    # Here, you'd need to play output_filename_fallback using playback.play_audio
                    # For simplicity with the current main.py structure, we'll assume direct play is primary.
                    # If playback.py is to be used, this method should just return the path to the real audio file.
                    # The current structure of SystemTTS implies it handles playback.
                    # For now, returning the fallback filename. The main app would need to play it.
                    # This makes SystemTTS's generate_speech a bit inconsistent.
                    # Let's stick to the "plays directly" model and fallback is just logged.
                    logger.warning("Fallback to file was generated but not automatically played by this simplified TTS method.")
                    # To actually use the fallback, main.py would need to check the return and play if it's a .wav
                    # For now, let's consider fallback as a "could have worked" scenario if direct failed.
                    # os.remove(output_filename_fallback) # Clean up if not used
                    return None # Sticking to the idea that generate_speech either plays or returns None.
                except Exception as e2:
                    logger.error(f"Fallback speech generation also failed: {str(e2)}")
            return None