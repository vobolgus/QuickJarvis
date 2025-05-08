"""
Audio recording functionality for the Voice Assistant.
This file remains largely the same, using PyAudio.
"""
import time
import wave
from contextlib import contextmanager
from typing import Generator, Optional, List
import os

import pyaudio

from utils import logger, get_timestamp

# Audio recording parameters
DEFAULT_FORMAT = pyaudio.paInt16  # 16-bit PCM
DEFAULT_CHANNELS = 1              # Mono
DEFAULT_RATE = 16000              # Whisper expects 16kHz audio
DEFAULT_CHUNK = 1024
DEFAULT_RECORD_SECONDS = 5
TEMP_FILE_DIR = "temp_recordings" # Store recordings in a sub-directory


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
                format_in: int = DEFAULT_FORMAT,
                channels_in: int = DEFAULT_CHANNELS,
                rate_in: int = DEFAULT_RATE,
                chunk_in: int = DEFAULT_CHUNK,
                input_device_index: Optional[int] = None) -> Generator[pyaudio.Stream, None, None]:
    """
    Context manager for handling an audio recording stream.

    Args:
        audio: PyAudio instance
        format_in: Audio format (default: paInt16)
        channels_in: Number of channels (default: 1)
        rate_in: Sample rate in Hz (default: 16000)
        chunk_in: Frames per buffer (default: 1024)
        input_device_index: Optional index of the input device.

    Yields:
        PyAudio Stream instance
    """
    stream = audio.open(
        format=format_in,
        channels=channels_in,
        rate=rate_in,
        input=True,
        frames_per_buffer=chunk_in,
        input_device_index=input_device_index
    )
    try:
        yield stream
    finally:
        stream.stop_stream()
        stream.close()


def record_audio(record_seconds: int = DEFAULT_RECORD_SECONDS,
                audio_format: int = DEFAULT_FORMAT,
                channels: int = DEFAULT_CHANNELS,
                rate: int = DEFAULT_RATE,
                chunk: int = DEFAULT_CHUNK,
                input_device_index: Optional[int] = None) -> str:
    """
    Record audio from microphone and save to a temporary file.

    Args:
        record_seconds: Duration of recording in seconds
        audio_format: Audio format (default: paInt16)
        channels: Number of channels (default: 1)
        rate: Sample rate in Hz (default: 16000)
        chunk: Frames per buffer (default: 1024)
        input_device_index: Optional specific microphone index.

    Returns:
        Path to the saved audio file
    """
    if not os.path.exists(TEMP_FILE_DIR):
        os.makedirs(TEMP_FILE_DIR)
        logger.info(f"Created directory for temporary recordings: {TEMP_FILE_DIR}")

    logger.info("Recording will start in 2 seconds...")
    time.sleep(2)

    frames: List[bytes] = []

    with audio_interface() as audio:
        if input_device_index is None:
            try:
                # Try to get default input device info for logging
                device_info = audio.get_default_input_device_info()
                logger.info(f"Using default input device: {device_info['name']}")
            except IOError:
                logger.warning("Could not get default input device info. Using system default.")
        else:
             try:
                device_info = audio.get_device_info_by_index(input_device_index)
                logger.info(f"Using specified input device: {device_info['name']} (Index: {input_device_index})")
             except IOError:
                logger.error(f"Invalid input device index: {input_device_index}. Falling back to default.")
                input_device_index = None # Fallback

        logger.info("Recording... Speak now!")
        with audio_stream(audio, audio_format, channels, rate, chunk, input_device_index) as stream:
            for _ in range(0, int(rate / chunk * record_seconds)):
                data = stream.read(chunk)
                frames.append(data)

        logger.info("Recording finished!")

        timestamp = get_timestamp()
        temp_filename = os.path.join(TEMP_FILE_DIR, f"recording_{timestamp}.wav")

        with wave.open(temp_filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(audio_format))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))

        logger.info(f"Audio saved to {temp_filename}")
        return temp_filename

def list_audio_devices():
    """Lists available audio input devices."""
    logger.info("Available audio input devices:")
    with audio_interface() as p:
        info = p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(0, numdevices):
            device_info = p.get_device_info_by_host_api_device_index(0, i)
            if (device_info.get('maxInputChannels')) > 0:
                logger.info(f"  Input Device ID {i} - {device_info.get('name')}")