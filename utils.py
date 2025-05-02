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
def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger with the specified log level.
    
    Args:
        level: The logging level (default: logging.INFO)
        
    Returns:
        A configured logger instance
    """
    # Create logger
    logger = logging.getLogger("voice_assistant")
    logger.setLevel(level)
    
    # Create console handler and set level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add handler to logger if not already added
    if not logger.handlers:
        logger.addHandler(console_handler)
    
    return logger

# Create the logger
logger = setup_logging()

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
        subprocess.check_output(['ffmpeg', '-version'], stderr=subprocess.STDOUT)
        logger.info("ffmpeg is installed and working.")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.error("Error: ffmpeg is not installed or not in PATH.")
        logger.info(
            "Please install ffmpeg:\n"
            "- On macOS: brew install ffmpeg\n"
            "- On Ubuntu/Debian: sudo apt install ffmpeg\n"
            "- On Windows: Download from https://www.ffmpeg.org/download.html and add to PATH"
        )
        return False

def get_device() -> Tuple[str, str]:
    """
    Determine the appropriate device (CPU/CUDA) and torch data type for model loading.
    
    Returns:
        A tuple containing the device type and torch data type as strings
    """
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = "float16" if torch.cuda.is_available() else "float32"
    
    logger.info(f"Device set to use {device} with dtype {torch_dtype}")
    return device, torch_dtype

def clean_temp_files(file_paths: list) -> None:
    """
    Remove temporary files if they exist.
    
    Args:
        file_paths: List of file paths to remove
    """
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Removed temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {file_path}: {e}")
