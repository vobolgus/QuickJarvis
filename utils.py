"""
Utility functions and common helpers for the Voice Assistant.
"""
import logging
import os
import subprocess
import sys
from datetime import datetime
from typing import Optional, Tuple, List # Added List

# Set up logging
def setup_logging(logger_level: int = logging.DEBUG, console_level: int = logging.WARNING) -> logging.Logger:
    """
    Configure and return a logger.

    Args:
        logger_level: The minimum level of messages the logger will process.
        console_level: The minimum level of messages that will be output to the console.

    Returns:
        A configured logger instance
    """
    current_logger = logging.getLogger("voice_assistant")
    current_logger.setLevel(logger_level) # Logger processes all messages at this level or higher

    # Prevent adding handlers multiple times if this function is called again
    if not current_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout) # Use stdout for handler
        console_handler.setLevel(console_level) # Console shows messages at this level or higher

        # Formatter for console messages (warnings, errors)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s') # Added module
        console_handler.setFormatter(formatter)
        current_logger.addHandler(console_handler)
    else:
        # If handlers exist, ensure console_level is updated if different
        for handler in current_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(console_level)

    return current_logger

# Create the logger: Process DEBUG and above, but console only shows WARNING and above by default
logger = setup_logging(logger_level=logging.DEBUG, console_level=logging.INFO) # Changed console to INFO for VAD messages

def get_timestamp() -> str:
    """
    Generate a timestamp string in the format YYYYMMDD_HHMMSS.

    Returns:
        A formatted timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def check_ffmpeg_installed() -> bool:
    """
    Check if ffmpeg is installed and available in the system PATH.

    Returns:
        True if ffmpeg is installed, False otherwise
    """
    try:
        # Use subprocess.run. capture_output=True handles stdout/stderr.
        # We don't need to specify stdout=DEVNULL or stderr=DEVNULL if we capture.
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True, # Capture stdout and stderr
            text=True,
            check=True # Raise CalledProcessError on non-zero exit
        )
        # Log a snippet of the version info for debugging, but it won't go to console by default
        if result.stdout:
            logger.debug(f"ffmpeg check successful. Version info snippet: {result.stdout.splitlines()[0]}")
        else:
            logger.debug("ffmpeg check successful (no version info in stdout).")
        return True
    except FileNotFoundError: # Specific error if ffmpeg command itself is not found
        logger.error("ffmpeg command not found. Please ensure it is installed and in your system's PATH.")
        print_ffmpeg_install_instructions()
        return False
    except subprocess.CalledProcessError as e: # ffmpeg found but exited with an error
        logger.error(f"ffmpeg command failed with exit code {e.returncode}.")
        if e.stderr:
            logger.error(f"ffmpeg stderr: {e.stderr.strip()}")
        print_ffmpeg_install_instructions() # Still relevant if it's misconfigured
        return False
    except (subprocess.SubprocessError, OSError) as e: # Other potential errors like permission issues
        logger.error(f"Error checking ffmpeg: {e}")
        print_ffmpeg_install_instructions()
        return False

def print_ffmpeg_install_instructions():
    """Prints ffmpeg installation instructions to the console."""
    print(
        "ERROR: ffmpeg is not installed, not found in PATH, or not working correctly.\n"
        "ffmpeg might be required by whisper.cpp for certain audio formats or processing.\n"
        "Please install/check ffmpeg:\n"
        "- On macOS (using Homebrew): brew install ffmpeg\n"
        "- On Ubuntu/Debian: sudo apt install ffmpeg\n"
        "- On Windows: Download from https://www.ffmpeg.org/download.html and add to PATH"
    )

def get_device() -> Tuple[str, str]:
    """
    Determine the appropriate device (CPU/CUDA) and torch data type for model loading.
    This function is not used by the current whisper.cpp setup but kept for potential future use.

    Returns:
        A tuple containing the device type and torch data type as strings
    """
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = "float16" if torch.cuda.is_available() else "float32"
        logger.debug(f"Torch device check: Using {device} with dtype {torch_dtype}")
        return device, torch_dtype
    except ImportError:
        logger.debug("PyTorch not found. get_device() returning default CPU setup.")
        return "cpu", "float32"
    except Exception as e:
        logger.warning(f"Error in get_device(): {e}. Defaulting to CPU.")
        return "cpu", "float32"


def clean_temp_files(file_paths: List[Optional[str]]) -> None:
    """
    Remove temporary files if they exist.

    Args:
        file_paths: List of file paths to remove (can contain None)
    """
    for file_path in file_paths:
        if file_path is None: # Skip if a None path was added (e.g., TTS failed)
            continue
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Removed temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {file_path}: {e}")

    # Clean up dummy TTS marker directory if it's empty
    tts_marker_dir = "temp_tts_markers"
    if os.path.exists(tts_marker_dir):
        try:
            if not os.listdir(tts_marker_dir): # Check if empty
                os.rmdir(tts_marker_dir)
                logger.debug(f"Removed empty temporary TTS marker directory: {tts_marker_dir}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary TTS marker directory {tts_marker_dir}: {e}")