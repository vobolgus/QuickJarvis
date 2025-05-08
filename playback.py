"""
Audio playback functionality for the Voice Assistant.
"""
import os
import subprocess
import sys
from typing import Optional

from utils import logger


def play_audio(file_path: str) -> bool:
    """
    Play an audio file using the system's default audio player.
    
    Args:
        file_path: Path to the audio file to play
        
    Returns:
        True if playback was successful, False otherwise
    """
    if not os.path.exists(file_path):
        logger.error(f"Audio file not found: {file_path}")
        return False
    
    logger.debug(f"Playing audio file: {file_path}")

    try:
        # Use platform-specific commands for audio playback
        if sys.platform == 'darwin':  # macOS
            subprocess.call(['afplay', file_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True

        elif sys.platform == 'linux':  # Linux
            subprocess.call(['aplay', file_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True

        elif sys.platform == 'win32':  # Windows
            # On Windows, use the start command to play the file with the default player
            # Suppressing output for 'start' is harder, os.system might show a window briefly.
            # For cleaner execution, a more specific Windows API call or library might be needed,
            # but 'start' is generally okay for simple cases.
            os.system(f'start /B {file_path}') # /B tries to avoid new window for some commands
            return True

        else:
            logger.warning(f"Unsupported OS for audio playback: {sys.platform}")
            return False

    except Exception as e:
        logger.error(f"Error playing audio: {str(e)}")
        return False