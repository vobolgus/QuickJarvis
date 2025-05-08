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
                input_device_index: Optional[int] = None) -> Generator[Optional[pyaudio.Stream], None, None]:
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
        PyAudio Stream instance, or None if stream opening fails
    """
    stream = None
    try:
        stream = audio.open(
            format=format_in,
            channels=channels_in,
            rate=rate_in,
            input=True,
            frames_per_buffer=chunk_in,
            input_device_index=input_device_index
        )
        yield stream
    except IOError as e:
        logger.error(f"Failed to open audio stream: {e}")
        if "Invalid sample rate" in str(e) or "Sample rate not supported" in str(e):
            logger.error(f"The sample rate {rate_in}Hz might not be supported by your microphone or system configuration.")
        yield None # Yield None if stream opening failed
    finally:
        if stream:
            try:
                if stream.is_active(): # Check if stream is active before stopping
                    stream.stop_stream()
                stream.close()
            except Exception as e_close:
                logger.warning(f"Error closing audio stream: {e_close}")


def record_audio(record_seconds: int = DEFAULT_RECORD_SECONDS,
                audio_format: int = DEFAULT_FORMAT,
                channels: int = DEFAULT_CHANNELS,
                rate: int = DEFAULT_RATE,
                chunk: int = DEFAULT_CHUNK,
                input_device_index: Optional[int] = None,
                suppress_prints: bool = False) -> Optional[str]:
    """
    Record audio from microphone and save to a temporary file.

    Args:
        record_seconds: Duration of recording in seconds
        audio_format: Audio format (default: paInt16)
        channels: Number of channels (default: 1)
        rate: Sample rate in Hz (default: 16000)
        chunk: Frames per buffer (default: 1024)
        input_device_index: Optional specific microphone index.
        suppress_prints: If True, suppress console messages like "Recording...".

    Returns:
        Path to the saved audio file, or None on failure.
    """
    if not os.path.exists(TEMP_FILE_DIR):
        try:
            os.makedirs(TEMP_FILE_DIR)
            logger.debug(f"Created directory for temporary recordings: {TEMP_FILE_DIR}")
        except OSError as e:
            logger.error(f"Could not create temp directory {TEMP_FILE_DIR}: {e}")
            return None

    if not suppress_prints:
        print("🎙️ Get ready to speak!") # Changed message and removed sleep
        # time.sleep(2) # REMOVED: Only pause if we're about to show "Recording..."

    frames: List[bytes] = []

    with audio_interface() as audio:
        if input_device_index is None:
            try:
                device_info = audio.get_default_input_device_info()
                logger.debug(f"Using default input device: {device_info['name']}")
            except IOError: # This can happen if no default input device is configured or available
                logger.warning("Could not get default input device info. Using system default, if any.")
        else:
             try:
                device_info = audio.get_device_info_by_index(input_device_index)
                logger.debug(f"Using specified input device: {device_info['name']} (Index: {input_device_index})")
             except IOError: # This can happen if the index is invalid
                logger.error(f"Invalid input device index: {input_device_index}. Falling back to default (if available).")
                input_device_index = None # Fallback

        if not suppress_prints:
            print("🔴 Recording... Speak now!")

        with audio_stream(audio, audio_format, channels, rate, chunk, input_device_index) as stream:
            if stream is None: # audio_stream yielded None due to an error
                logger.error("Audio stream could not be opened. Recording aborted.")
                return None

            num_chunks_to_record = int(rate / chunk * record_seconds)
            if num_chunks_to_record <= 0: # Ensure we record at least one chunk if duration is very short
                num_chunks_to_record = 1
                logger.warning(f"Record duration {record_seconds}s is very short, recording {num_chunks_to_record} chunk(s).")

            try:
                for i in range(0, num_chunks_to_record):
                    data = stream.read(chunk, exception_on_overflow=False)
                    frames.append(data)
            except IOError as e: # PyAudioError can be an IOError subclass
                 logger.error(f"Error reading from audio stream: {e}. Recording may be incomplete.")
                 # Proceed with what was recorded if any, or return None if critical
                 if not frames:
                     return None


        if not suppress_prints:
            print("✅ Recording finished!")

        if not frames: # No data recorded
            logger.error("No audio data was recorded.")
            return None

        timestamp = get_timestamp()
        temp_filename = os.path.join(TEMP_FILE_DIR, f"recording_{timestamp}.wav")

        try:
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(audio.get_sample_size(audio_format))
                wf.setframerate(rate)
                wf.writeframes(b''.join(frames))
        except Exception as e:
            logger.error(f"Failed to save WAV file {temp_filename}: {e}")
            return None

        logger.debug(f"Audio saved to {temp_filename}")
        return temp_filename

def list_audio_devices():
    """Lists available audio input devices."""
    print("\nAvailable audio input devices:")
    try:
        with audio_interface() as p:
            # Try to get info for the default host API first
            try:
                default_host_api_index = p.get_default_host_api_info()['index']
            except Exception: # Fallback if default host API info fails
                default_host_api_index = 0 # Assume 0 as a common default

            host_api_info = p.get_host_api_info_by_index(default_host_api_index)
            numdevices = host_api_info.get('deviceCount', 0)

            found_devices = False
            for i in range(0, numdevices):
                try:
                    device_info = p.get_device_info_by_host_api_device_index(default_host_api_index, i)
                    if (device_info.get('maxInputChannels')) > 0:
                        print(f"  Input Device ID {i} - {device_info.get('name')}")
                        found_devices = True
                except Exception as e_dev:
                    logger.debug(f"Could not get info for device index {i} on host API {default_host_api_index}: {e_dev}")

            if not found_devices:
                # If no devices found on default host API, try iterating all host APIs
                logger.debug("No input devices on default host API, checking all host APIs...")
                num_host_apis = p.get_host_api_count()
                for host_api_idx in range(num_host_apis):
                    if host_api_idx == default_host_api_index: continue # Skip already checked
                    try:
                        host_api_info = p.get_host_api_info_by_index(host_api_idx)
                        numdevices_alt = host_api_info.get('deviceCount', 0)
                        for i_alt in range(numdevices_alt):
                            device_info_alt = p.get_device_info_by_host_api_device_index(host_api_idx, i_alt)
                            if (device_info_alt.get('maxInputChannels')) > 0:
                                print(f"  Input Device ID {device_info_alt.get('index')} ({host_api_info.get('name')}) - {device_info_alt.get('name')}")
                                found_devices = True
                    except Exception as e_host:
                        logger.debug(f"Could not enumerate devices for host API index {host_api_idx}: {e_host}")

            if not found_devices:
                print("  No input devices found.")
    except Exception as e:
        logger.error(f"Could not list audio devices: {e}")
        print("  Error listing audio devices.")
    print("")