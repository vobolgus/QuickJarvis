"""
Audio playback functionality for the Voice Assistant.
Enhanced with improved error handling and fallback mechanisms.
Python 3.10 compatible with marker file handling.
"""
import os
import subprocess
import sys
import platform
import time
from typing import Optional, List

from utils import logger

def play_audio(file_path: str) -> bool:
    """
    Play an audio file using the system's default audio player.
    Enhanced with fallback methods and improved error handling.

    Args:
        file_path: Path to the audio file to play

    Returns:
        True if playback was successful, False otherwise
    """
    if not os.path.exists(file_path):
        logger.error(f"Audio file not found: {file_path}")
        return False

    # Handle marker files - these are not real audio files but indicators that TTS was done directly
    if file_path.endswith('.marker') or 'marker' in file_path:
        # Check file content to confirm it's a marker
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(100)
                if "Marker for" in content or "TTS marker" in content:
                    logger.debug(f"Detected marker file, speech already played: {file_path}")
                    return True  # Speech was already played, no need to play again
        except Exception as e:
            logger.warning(f"Error checking marker file: {e}")

    # Check if the file is empty or invalid
    if os.path.getsize(file_path) == 0:
        logger.error(f"Audio file is empty: {file_path}")
        return False

    # Some text files might be created as placeholders - check for this
    if os.path.getsize(file_path) < 1000:  # Small files are suspicious
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(100)
                if any(marker in content for marker in ["Fallback TTS marker", "TTS fallback", "Marker for"]):
                    logger.warning(f"This is a marker file, not an actual audio file: {file_path}")
                    # For marker files, return success since the sound was already played
                    return True
        except:
            pass

    logger.debug(f"Playing audio file: {file_path}")

    # Determine platform
    platform_name = platform.system().lower()
    success = False
    error_message = None

    try:
        # Try each platform with multiple fallback methods
        if platform_name == 'darwin':  # macOS
            # Method 1: afplay (standard macOS audio player)
            try:
                logger.debug("Trying macOS afplay...")
                subprocess.run(['afplay', file_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                success = True
            except Exception as e:
                error_message = f"afplay failed: {str(e)}"
                logger.warning(error_message)

                # Method 2: Try using 'open' to open with the default app
                try:
                    logger.debug("Trying macOS open...")
                    subprocess.run(['open', file_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    success = True
                except Exception as e2:
                    error_message += f", open failed: {str(e2)}"
                    logger.warning(f"open failed: {str(e2)}")

        elif platform_name == 'linux':  # Linux
            # Method 1: aplay (standard Linux audio player)
            try:
                logger.debug("Trying Linux aplay...")
                subprocess.run(['aplay', '-q', file_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                success = True
            except Exception as e:
                error_message = f"aplay failed: {str(e)}"
                logger.warning(error_message)

                # Method 2: Try using 'play' from SoX
                try:
                    logger.debug("Trying Linux play (SoX)...")
                    subprocess.run(['play', '-q', file_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    success = True
                except Exception as e2:
                    error_message += f", play failed: {str(e2)}"
                    logger.warning(f"play failed: {str(e2)}")

                    # Method 3: Try using 'paplay' from PulseAudio
                    try:
                        logger.debug("Trying Linux paplay (PulseAudio)...")
                        subprocess.run(['paplay', file_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                        success = True
                    except Exception as e3:
                        error_message += f", paplay failed: {str(e3)}"
                        logger.warning(f"paplay failed: {str(e3)}")

        elif platform_name == 'win32' or platform_name == 'windows':  # Windows
            # Method 1: Use the 'start' command (Windows shell)
            try:
                logger.debug("Trying Windows start command...")
                os.system(f'start /MIN "" "{os.path.abspath(file_path)}"')
                success = True
                # Give Windows a moment to launch the player
                time.sleep(0.5)
            except Exception as e:
                error_message = f"Windows start failed: {str(e)}"
                logger.warning(error_message)

                # Method 2: Try PowerShell for playback
                try:
                    logger.debug("Trying Windows PowerShell for audio playback...")
                    # Fix for Python 3.10 - escape backslashes outside the f-string
                    abs_path = os.path.abspath(file_path)
                    escaped_path = abs_path.replace("\\", "\\\\")
                    ps_command = f'(New-Object Media.SoundPlayer "{escaped_path}").PlaySync();'
                    subprocess.run(['powershell', '-Command', ps_command], check=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    success = True
                except Exception as e2:
                    error_message += f", PowerShell playback failed: {str(e2)}"
                    logger.warning(f"PowerShell playback failed: {str(e2)}")

        else:
            logger.warning(f"Unsupported OS for audio playback: {platform_name}")
            success = False
            error_message = f"Unsupported OS: {platform_name}"

        # Force direct playback as last resort
        if not success:
            try:
                logger.warning(f"All standard playback methods failed. Forcing playback with Python...")

                # This part requires the audio module which is imported conditionally
                # to avoid requiring it if standard methods work
                try:
                    # Try using simpleaudio if available (good cross-platform support)
                    import simpleaudio as sa
                    logger.debug("Using simpleaudio for fallback playback...")
                    wave_obj = sa.WaveObject.from_wave_file(file_path)
                    play_obj = wave_obj.play()
                    play_obj.wait_done()
                    success = True
                except ImportError:
                    # Try pydub if available
                    try:
                        from pydub import AudioSegment
                        from pydub.playback import play
                        logger.debug("Using pydub for fallback playback...")
                        sound = AudioSegment.from_file(file_path)
                        play(sound)
                        success = True
                    except ImportError:
                        logger.warning("Neither simpleaudio nor pydub available for fallback playback")
            except Exception as e_fallback:
                error_message += f", Fallback playback failed: {str(e_fallback)}"
                logger.error(f"Fallback audio playback also failed: {str(e_fallback)}")

        if success:
            logger.debug("Audio playback completed successfully")
            return True
        else:
            logger.error(f"All audio playback methods failed: {error_message}")
            return False

    except Exception as e:
        logger.error(f"Unexpected error playing audio: {str(e)}")
        return False

def play_audio_list(file_paths: List[str]) -> bool:
    """
    Play a list of audio files in sequence.

    Args:
        file_paths: List of paths to audio files to play

    Returns:
        True if all playbacks were successful, False if any failed
    """
    all_successful = True

    for file_path in file_paths:
        if file_path is None:
            continue

        success = play_audio(file_path)
        if not success:
            all_successful = False

    return all_successful