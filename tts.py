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
        logger.debug(f"Available voices: {self.available_voices if self.available_voices else 'None found or not queried/supported'}")

        self.voice = voice or self._get_default_voice(language)
        if self.voice:
            logger.debug(f"Using voice: {self.voice}")
        else:
            logger.debug(f"No specific voice set; will use system default for language '{language}'.")


    def _get_platform(self) -> str:
        """Determine the operating system platform."""
        import platform
        return platform.system().lower()

    def _get_default_voice(self, language: str) -> str:
        """Get default voice based on language and availability."""
        if not self.available_voices: # if list is empty or None
            return "" # Rely on system's language default if no voices listed/found

        lang_lower = language.lower()
        # Try to find language-appropriate voice
        if self.platform == "darwin": # macOS has named voices
            # More robust matching could involve checking language codes in 'say -v ?' output
            if lang_lower.startswith("ru"):
                russian_voices = ["Yuri", "Milena", "Katya"] # Common names
                for v_name in russian_voices:
                    if v_name in self.available_voices: return v_name
            elif lang_lower.startswith("en"):
                english_voices = ["Samantha", "Alex", "Victoria", "Daniel", "Karen", "Tessa", "Fred"] # Common names
                for v_name in english_voices:
                    if v_name in self.available_voices: return v_name

            # Generic fallback: if a voice name contains the lang code (e.g., "en_US")
            for v_name in self.available_voices:
                if f"_{lang_lower.split('-')[0]}" in v_name.lower() or f" {lang_lower.split('-')[0]}" in v_name.lower():
                    return v_name


        # For Linux (espeak) and Windows, voice selection is often by language code directly or system default.
        # If a specific voice name matching logic is needed for other platforms, add here.

        # If no specific match, return the first available voice as a last resort, or empty string to use system default.
        # return self.available_voices[0] if self.available_voices else ""
        return "" # Prefer system default if no specific match

    def get_available_voices(self) -> List[str]:
        """Get list of available voices on the system (macOS specific for now)."""
        voices = []
        if self.platform == "darwin":
            try:
                # 'say -v ?' output format: VoiceName LangCode # Comment
                # Example: Alex                en_US    # Most people recognize me by my voice.
                result = subprocess.run(['say', '-v', '?'],
                                       capture_output=True, text=True, check=True, encoding='utf-8')
                for line in result.stdout.splitlines():
                    if line.strip() and not line.startswith('#'): # Ignore empty lines and comments
                        parts = line.split('#', 1)[0].strip() # Get part before comment
                        voice_name_and_lang = parts.split(None, 1) # Split voice name from lang code
                        if voice_name_and_lang and voice_name_and_lang[0]:
                            voices.append(voice_name_and_lang[0])
                logger.debug(f"Found {len(voices)} voices on macOS: {voices}")
                return voices
            except subprocess.SubprocessError as e:
                logger.error(f"Error getting voices on macOS: {str(e)}")
                return []
        # Placeholder for other platforms if voice listing is implemented
        # elif self.platform == "linux":
        #   try: # espeak --voices | grep '^[a-z]' ...
        # elif self.platform == "windows":
        #   try: # PowerShell: Get-SpeechVoice | Select-Object -ExpandProperty Name
        else:
            logger.debug("Getting available voices not implemented for this platform or no voices found.")
            return []

    def generate_speech(self, text: str, **kwargs) -> Optional[str]:
        """
        Convert text to speech using direct system commands.
        Creates a dummy marker file for compatibility if speech is played directly.

        Args:
            text: Text to convert to speech

        Returns:
            Path to a dummy marker file if successful, None otherwise.
        """
        if not text.strip():
            logger.debug("Skipping TTS for empty text.")
            return None

        try:
            timestamp = get_timestamp()
            # Dummy marker filename, as speech is played directly.
            # This helps with the `clean_temp_files` logic if needed.
            output_marker_filename = f"temp_speech_played_{timestamp}.marker"

            logger.debug(f"Generating direct speech for text (first 50 chars): \"{text[:50]}...\"")
            start_time = time.time()
            success = False

            # Basic sanitization for command line. More robust methods might be needed for complex text.
            # For 'say' and 'espeak', direct passing is often fine. PowerShell is more sensitive.
            # `subprocess.run` with a list of arguments (not `shell=True`) handles most spacing issues.

            if self.platform == "darwin":
                cmd = ['say']
                if self.voice and self.voice in self.available_voices:
                    cmd.extend(['-v', self.voice])
                # Language can also be specified with -r for rate, etc.
                # cmd.extend(['--language', self.language]) # 'say' uses voice's language
                cmd.append(text)
                logger.debug(f"Running macOS 'say' command: {cmd}")
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                success = True

            elif self.platform == "linux":
                cmd = ['espeak']
                # espeak voice selection can be complex (lang+variant). -v <lang> is common.
                if self.voice: # if a specific espeak voice name/path is known
                    cmd.extend(['-v', self.voice])
                elif self.language:
                     cmd.extend(['-v', self.language.split('-')[0]]) # e.g., "en" from "en-US"
                # cmd.extend(['--stdin']) # Alternative: pass text via stdin
                cmd.append(text)
                logger.debug(f"Running Linux 'espeak' command: {cmd}")
                try:
                    # espeak might output to stderr even on success for info, so capture and check return code
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    if result.stderr and result.stderr.strip():
                        logger.debug(f"espeak stderr: {result.stderr.strip()}")
                    success = True
                except (subprocess.SubprocessError, FileNotFoundError) as e:
                    logger.error(f"Failed to use espeak: {e}. Is it installed and in PATH? (e.g., sudo apt-get install espeak)")


            elif self.platform == "windows":
                # PowerShell needs careful quoting for the text.
                ps_safe_text = text.replace('"', '`"').replace("'", "''") # Escape " for PS, and ' for SQL-like within PS

                ps_script = "Add-Type -AssemblyName System.Speech;\n"
                ps_script += "$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer;\n"
                if self.voice: # Assumes self.voice is a valid Windows voice name from Get-SpeechVoice
                     ps_script += f'$speak.SelectVoice("{self.voice}");\n'
                # Note: Setting language directly on SpeechSynthesizer is not straightforward.
                # It usually picks based on the selected voice or system's language settings.
                ps_script += f'$speak.Speak("{ps_safe_text}");'

                logger.debug(f"Running Windows PowerShell speech command...")
                # Use -Command - for script block, or save to .ps1 and execute
                # Powershell execution policy might be an issue for some users.
                subprocess.run(
                    ['powershell', '-ExecutionPolicy', 'Unrestricted', '-Command', ps_script],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
                )
                success = True
            else:
                logger.error(f"TTS not supported on platform: {self.platform}")

            end_time = time.time()

            if success:
                logger.debug(f"Direct speech synthesis completed in {end_time - start_time:.2f} seconds")
                # Create a dummy marker file for cleanup logic compatibility
                with open(output_marker_filename, 'w') as f:
                    f.write(f"Marker for directly played TTS audio at {timestamp}.")
                return output_marker_filename
            else:
                logger.error("Direct speech synthesis failed or was not attempted for this platform.")
                return None

        except subprocess.CalledProcessError as e:
            # Log stderr from the failed process if available
            stderr_output = e.stderr.decode('utf-8', errors='ignore').strip() if e.stderr else "N/A"
            logger.error(f"TTS command failed with exit code {e.returncode}. Stderr: {stderr_output}", exc_info=False) # exc_info=False as we logged stderr
            return None
        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}", exc_info=True)
            return None