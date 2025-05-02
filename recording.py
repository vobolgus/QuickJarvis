"""
Audio recording functionality for the Voice Assistant.
"""
import time
import wave
from contextlib import contextmanager
from typing import Generator, Optional, List

import pyaudio

from utils import logger, get_timestamp

# Audio recording parameters
DEFAULT_FORMAT = pyaudio.paInt16
DEFAULT_CHANNELS = 1
DEFAULT_RATE = 16000  # Whisper expects 16kHz audio
DEFAULT_CHUNK = 1024
DEFAULT_RECORD_SECONDS = 5


@contextmanager
def audio_interface() -> Generator[pyaudio.PyAudio, None, None]:
    """
    Context manager for initializing and cleaning up PyAudio interface.
    
    Yields:
        PyAudio interface instance
    """
    audio = pyaudio.PyAudio()
    try:
        yield audio
    finally:
        audio.terminate()


@contextmanager
def audio_stream(audio: pyaudio.PyAudio, 
                format: int = DEFAULT_FORMAT,
                channels: int = DEFAULT_CHANNELS,
                rate: int = DEFAULT_RATE,
                chunk: int = DEFAULT_CHUNK) -> Generator[pyaudio.Stream, None, None]:
    """
    Context manager for handling an audio recording stream.
    
    Args:
        audio: PyAudio instance
        format: Audio format (default: paInt16)
        channels: Number of channels (default: 1)
        rate: Sample rate in Hz (default: 16000)
        chunk: Frames per buffer (default: 1024)
        
    Yields:
        PyAudio Stream instance
    """
    stream = audio.open(
        format=format,
        channels=channels,
        rate=rate,
        input=True,
        frames_per_buffer=chunk
    )
    try:
        yield stream
    finally:
        stream.stop_stream()
        stream.close()


def record_audio(record_seconds: int = DEFAULT_RECORD_SECONDS,
                format: int = DEFAULT_FORMAT,
                channels: int = DEFAULT_CHANNELS,
                rate: int = DEFAULT_RATE,
                chunk: int = DEFAULT_CHUNK) -> str:
    """
    Record audio from microphone and save to a temporary file.
    
    Args:
        record_seconds: Duration of recording in seconds
        format: Audio format (default: paInt16)
        channels: Number of channels (default: 1)
        rate: Sample rate in Hz (default: 16000)
        chunk: Frames per buffer (default: 1024)
        
    Returns:
        Path to the saved audio file
    """
    logger.info("Recording will start in 2 seconds...")
    time.sleep(2)
    
    frames: List[bytes] = []
    
    # Use context managers for PyAudio and stream
    with audio_interface() as audio:
        logger.info("Recording... Speak now!")
        with audio_stream(audio, format, channels, rate, chunk) as stream:
            # Collect audio data
            for _ in range(0, int(rate / chunk * record_seconds)):
                data = stream.read(chunk)
                frames.append(data)
        
        logger.info("Recording finished!")
        
        # Save the recorded audio to a temporary file
        timestamp = get_timestamp()
        temp_filename = f"temp_recording_{timestamp}.wav"
        
        # Save the audio file using context manager
        with wave.open(temp_filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(format))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))
        
        logger.info(f"Audio saved to {temp_filename}")
        return temp_filename
