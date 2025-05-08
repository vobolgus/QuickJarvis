"""
Utility functions and common helpers for the Voice Assistant.
"""
import logging
import os
import subprocess
import sys
from datetime import datetime
from typing import Optional, Tuple

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
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        current_logger.addHandler(console_handler)
    else:
        # If handlers exist, ensure console_level is updated if different
        for handler in current_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(console_level)

    return current_logger

# Create the logger: Process DEBUG and above, but console only shows WARNING and above by default
logger = setup_logging(logger_level=logging.DEBUG, console_level=logging.WARNING)

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
        # Use subprocess.run for better control and modern Python
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True)
        logger.debug(f"ffmpeg check successful. Version info snippet: {result.stdout.splitlines()[0]}")
        return True
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.error(f"Error checking ffmpeg: {e}")
        # Use print for user-facing critical setup issue
        print(
            "ERROR: ffmpeg is not installed or not found in PATH.\n"
            "ffmpeg is required for audio processing.\n"
            "Please install ffmpeg:\n"
            "- On macOS (using Homebrew): brew install ffmpeg\n"
            "- On Ubuntu/Debian: sudo apt install ffmpeg\n"
            "- On Windows: Download from https://www.ffmpeg.org/download.html and add to PATH"
        )
        return False

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


def clean_temp_files(file_paths: list) -> None:
    """
    Remove temporary files if they exist.

    Args:
        file_paths: List of file paths to remove
    """
    for file_path in file_paths:
        if file_path is None: # Skip if a None path was added
            continue
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Removed temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {file_path}: {e}")