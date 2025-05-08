"""
Audio recording functionality for the Voice Assistant.
This file remains largely the same, using PyAudio.
"""
import time
import wave
from contextlib import contextmanager
from typing import Generator, Optional, List, Deque
import os
import collections

import pyaudio
import webrtcvad # For Voice Activity Detection

from utils import logger, get_timestamp

# Audio recording parameters
DEFAULT_FORMAT = pyaudio.paInt16  # 16-bit PCM
DEFAULT_CHANNELS = 1              # Mono
DEFAULT_RATE = 16000              # Whisper expects 16kHz audio, VAD also supports this
DEFAULT_CHUNK_FIXED = 1024        # Default chunk size for fixed duration recording
DEFAULT_RECORD_SECONDS = 5
TEMP_FILE_DIR = "temp_recordings" # Store recordings in a sub-directory

# VAD constants
VAD_AGGRESSIVENESS = 1  # 0 (least aggressive) to 3 (most aggressive)
VAD_FRAME_MS = 30       # ms, webrtcvad supports 10, 20, 30. This will be the chunk duration for VAD.
VAD_PADDING_MS = 300    # ms of silence to keep before and after speech
VAD_SILENCE_TIMEOUT_MS = 1000 # ms of silence after speech to stop recording
VAD_MIN_SPEECH_MS = 250 # ms, minimum duration of speech to consider valid


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
                pyaudio_frames_per_buffer: int = DEFAULT_CHUNK_FIXED,
                input_device_index: Optional[int] = None) -> Generator[Optional[pyaudio.Stream], None, None]:
    """
    Context manager for handling an audio recording stream.

    Args:
        audio: PyAudio instance
        format_in: Audio format (default: paInt16)
        channels_in: Number of channels (default: 1)
        rate_in: Sample rate in Hz (default: 16000)
        pyaudio_frames_per_buffer: PyAudio frames per buffer (chunk size for PyAudio stream)
        input_device_index: Optional index of the input device.

    Yields:
        PyAudio Stream instance, or None if stream opening fails
    """
    stream = None
    try:
        logger.debug(f"Opening audio stream with: Rate={rate_in}, Channels={channels_in}, Format={format_in}, PyAudioFramesPerBuffer={pyaudio_frames_per_buffer}, DeviceIdx={input_device_index}")
        stream = audio.open(
            format=format_in,
            channels=channels_in,
            rate=rate_in,
            input=True,
            frames_per_buffer=pyaudio_frames_per_buffer,
            input_device_index=input_device_index
        )
        yield stream
    except IOError as e:
        logger.error(f"Failed to open audio stream: {e}")
        if "Invalid sample rate" in str(e) or "Sample rate not supported" in str(e):
            logger.error(f"The sample rate {rate_in}Hz might not be supported by your microphone or system configuration.")
        if "Invalid input device" in str(e):
            logger.error(f"Invalid input device index ({input_device_index}). Please check available devices with --list-devices.")
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
                input_device_index: Optional[int] = None,
                suppress_prints: bool = False,
                use_vad: bool = False) -> Optional[str]:
    """
    Record audio from microphone. Uses fixed duration or VAD based on `use_vad`.

    Args:
        record_seconds: Duration for fixed recording, or max_duration for VAD.
        audio_format: Audio format.
        channels: Number of channels.
        rate: Sample rate in Hz.
        input_device_index: Specific microphone index.
        suppress_prints: If True, suppress console messages.
        use_vad: If True, use Voice Activity Detection.

    Returns:
        Path to the saved audio file, or None on failure or if no speech detected with VAD.
    """
    if not os.path.exists(TEMP_FILE_DIR):
        try:
            os.makedirs(TEMP_FILE_DIR)
            logger.debug(f"Created directory for temporary recordings: {TEMP_FILE_DIR}")
        except OSError as e:
            logger.error(f"Could not create temp directory {TEMP_FILE_DIR}: {e}")
            return None

    frames_data: List[bytes] = [] # Holds the final audio data to be saved
    pyaudio_chunk_size_to_use: int # PyAudio frames per buffer

    with audio_interface() as audio:
        sample_width = audio.get_sample_size(audio_format)

        if input_device_index is None:
            try:
                device_info = audio.get_default_input_device_info()
                logger.debug(f"Using default input device: {device_info['name']}")
            except IOError:
                logger.warning("Could not get default input device info. Using system default, if any.")
        else:
             try:
                device_info = audio.get_device_info_by_index(input_device_index)
                logger.debug(f"Using specified input device: {device_info['name']} (Index: {input_device_index})")
             except IOError:
                logger.error(f"Invalid input device index: {input_device_index}. Falling back to default (if available).")
                input_device_index = None

        if use_vad:
            vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

            # Number of PyAudio frames for one VAD frame duration (e.g., 30ms)
            pyaudio_frames_per_vad_buffer = (rate // 1000) * VAD_FRAME_MS # e.g., 16 * 30 = 480 PyAudio frames
            pyaudio_chunk_size_to_use = pyaudio_frames_per_vad_buffer

            # Byte length of one VAD audio frame (data passed to VAD.is_speech)
            vad_frame_byte_length = pyaudio_frames_per_vad_buffer * channels * sample_width # e.g., 480 * 1 * 2 = 960 bytes

            if pyaudio_frames_per_vad_buffer == 0:
                logger.error("Calculated VAD PyAudio frames per buffer is zero. Check rate, VAD_FRAME_MS.")
                return None
            logger.debug(f"VAD Mode: PyAudio frames per buffer={pyaudio_frames_per_vad_buffer} ({VAD_FRAME_MS}ms), VAD frame data length={vad_frame_byte_length} bytes")

            num_padding_frames = VAD_PADDING_MS // VAD_FRAME_MS
            ring_buffer: Deque[bytes] = collections.deque(maxlen=num_padding_frames)

            triggered = False
            voiced_count = 0
            silence_count = 0

            # Max number of VAD buffers to process before timeout
            max_vad_buffers_to_process = int((record_seconds * 1000) / VAD_FRAME_MS)
            min_speech_vad_frames = VAD_MIN_SPEECH_MS // VAD_FRAME_MS
            silence_timeout_frames = VAD_SILENCE_TIMEOUT_MS // VAD_FRAME_MS

            if not suppress_prints:
                print("👂 Listening for speech...")

            with audio_stream(audio, audio_format, channels, rate, pyaudio_chunk_size_to_use, input_device_index) as stream:
                if stream is None:
                    logger.error("Audio stream could not be opened for VAD. Recording aborted.")
                    return None

                for _ in range(max_vad_buffers_to_process): # Loop for timeout
                    try:
                        # Read one VAD frame's worth of audio data (in PyAudio frames)
                        frame_bytes = stream.read(pyaudio_frames_per_vad_buffer, exception_on_overflow=False)
                    except IOError as e:
                        logger.error(f"Error reading from audio stream during VAD: {e}")
                        break

                    if not frame_bytes or len(frame_bytes) != vad_frame_byte_length:
                        logger.warning(f"Read unexpected data length from audio stream ({len(frame_bytes) if frame_bytes else 'None'} bytes, expected {vad_frame_byte_length}). Ending VAD.")
                        break

                    is_speech = vad.is_speech(frame_bytes, rate)

                    if not triggered:
                        ring_buffer.append(frame_bytes)
                        if is_speech:
                            if not suppress_prints:
                                print("🎤 Speech detected, recording...")
                            triggered = True
                            frames_data.extend(list(ring_buffer))
                            voiced_count = 1
                            silence_count = 0
                    else:
                        frames_data.append(frame_bytes)
                        if is_speech:
                            voiced_count += 1
                            silence_count = 0
                        else:
                            silence_count += 1
                            if silence_count > silence_timeout_frames:
                                if not suppress_prints:
                                    print("🛑 Silence detected, finishing recording.")
                                break
                else:
                    if triggered and not suppress_prints:
                        print("⌛ Recording timed out.")
                    elif not triggered and not suppress_prints:
                        print("⌛ Listening timed out, no speech detected.")

            if not triggered or voiced_count < min_speech_vad_frames:
                if not suppress_prints:
                    if triggered: # Was triggered but speech too short
                         print(f"🎙️ Recording too short ({voiced_count * VAD_FRAME_MS}ms), discarded.")
                    # else: "Listening timed out, no speech detected" already printed
                logger.info(f"VAD: No significant speech detected or recording too short (voiced VAD frames: {voiced_count}). Discarding.")
                return None

        else: # Fixed duration recording
            pyaudio_chunk_size_to_use = DEFAULT_CHUNK_FIXED
            logger.debug(f"Fixed Duration Mode: PyAudio frames per buffer={pyaudio_chunk_size_to_use}")

            if not suppress_prints:
                print(f"🔴 Recording for {record_seconds}s... Speak now!")

            with audio_stream(audio, audio_format, channels, rate, pyaudio_chunk_size_to_use, input_device_index) as stream:
                if stream is None:
                    logger.error("Audio stream could not be opened for fixed recording. Recording aborted.")
                    return None

                num_chunks_to_record_fixed = int(rate / pyaudio_chunk_size_to_use * record_seconds)
                if num_chunks_to_record_fixed <= 0:
                    num_chunks_to_record_fixed = 1
                    logger.warning(f"Record duration {record_seconds}s is very short, recording {num_chunks_to_record_fixed} chunk(s).")

                for _ in range(num_chunks_to_record_fixed):
                    try:
                        data = stream.read(pyaudio_chunk_size_to_use, exception_on_overflow=False)
                        frames_data.append(data)
                    except IOError as e:
                         logger.error(f"Error reading from audio stream: {e}. Recording may be incomplete.")
                         if not frames_data: return None
                         break

            if not suppress_prints:
                print("✅ Recording finished!")

        if not frames_data:
            logger.error("No audio data was recorded.")
            return None

        timestamp = get_timestamp()
        temp_filename = os.path.join(TEMP_FILE_DIR, f"recording_{timestamp}.wav")

        try:
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(rate)
                wf.writeframes(b''.join(frames_data))
        except Exception as e:
            logger.error(f"Failed to save WAV file {temp_filename}: {e}")
            return None

        total_bytes_recorded = sum(len(f) for f in frames_data)
        logger.debug(f"Audio saved to {temp_filename} ({total_bytes_recorded} bytes)")
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