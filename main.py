"""
Voice Assistant main application that integrates speech recognition,
text analysis, and speech synthesis components.
"""
import os
import sys
import traceback
from typing import List, Optional

# Import modules
from utils import logger, clean_temp_files, check_ffmpeg_installed # get_device removed as not needed for whisper.cpp
from recording import record_audio, DEFAULT_RECORD_SECONDS, list_audio_devices # Added list_audio_devices and DEFAULT_RECORD_SECONDS for consistency if needed
from transcription_cpp import WhisperCppTranscriber # CHANGED: Using WhisperCppTranscriber
from analysis import GemmaAnalyzer
from playback import play_audio
from tts import SystemTTS

# --- Configuration for whisper.cpp ---
# IMPORTANT: Adjust these paths to your whisper.cpp installation and model
# You can also set these as environment variables: WHISPER_CPP_DIR, WHISPER_CLI_PATH, WHISPER_MODEL_PATH

# Path to your local clone of the whisper.cpp repository
# Example: WHISPER_CPP_DIR = "/Users/yourusername/dev/whisper.cpp"
DEFAULT_WHISPER_CPP_DIR = os.path.expanduser("~/dev/whisper.cpp")
WHISPER_CPP_DIR = os.getenv("WHISPER_CPP_DIR", DEFAULT_WHISPER_CPP_DIR)

# Path to the whisper-cli executable within the whisper.cpp directory
DEFAULT_WHISPER_CLI_PATH = os.path.join(WHISPER_CPP_DIR, "build/bin/whisper-cli") # Adjusted for common build structure
WHISPER_CLI_PATH = os.getenv("WHISPER_CLI_PATH", DEFAULT_WHISPER_CLI_PATH)

# Name of the ggml model file (e.g., "ggml-base.en.bin", "ggml-small.en.bin")
# This file should be in the WHISPER_CPP_DIR/models/ directory
MODEL_FILENAME = "ggml-base.en.bin"  # Recommended: start with base.en, or use medium/large for better accuracy
# MODEL_FILENAME = "ggml-large-v3.bin" # For higher accuracy if downloaded and built for it

DEFAULT_MODEL_PATH = os.path.join(WHISPER_CPP_DIR, "models", MODEL_FILENAME)
MODEL_PATH = os.getenv("WHISPER_MODEL_PATH", DEFAULT_MODEL_PATH)

# Recording duration (can be taken from recording.py or defined here)
RECORD_SECONDS = DEFAULT_RECORD_SECONDS

# Specify microphone (Optional). If None, uses default.
# Run with --list-devices to see available microphone IDs.
MICROPHONE_INDEX = None  # Example: set to 0 or 1 if you have multiple mics
# --- End Configuration ---


def main() -> int:
    """
    Main function that runs the Voice Assistant application.

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    logger.info("\nVoice Assistant with whisper.cpp, Gemma and System TTS") # UPDATED
    logger.info("----------------------------------------------------")
    logger.info("This program will:")
    logger.info("1. Record your voice and transcribe it with whisper.cpp") # UPDATED
    logger.info("2. Analyze the transcription with Gemma")
    logger.info("3. Read the analysis back to you using System TTS") # UPDATED
    logger.info(f"Recording duration: {RECORD_SECONDS} seconds")

    logger.info(f"Using WHISPER_CPP_DIR: {WHISPER_CPP_DIR}")
    logger.info(f"Using WHISPER_CLI_PATH: {WHISPER_CLI_PATH}")
    logger.info(f"Using MODEL_PATH: {MODEL_PATH}")

    if "--list-devices" in sys.argv:
        list_audio_devices() # Using the function from recording.py
        return 0

    # Check prerequisites
    if not check_ffmpeg_installed(): # ffmpeg might still be useful for whisper.cpp or general audio handling
        return 1

    # Get device configuration - Not needed for whisper.cpp transcriber
    # device, _ = get_device() # REMOVED

    # Initialize components
    try:
        # Initialize Whisper.cpp model for transcription
        logger.info("Initializing whisper.cpp transcriber...")
        try:
            transcriber = WhisperCppTranscriber(
                model_path=MODEL_PATH,
                whisper_cli_path=WHISPER_CLI_PATH
            )
        except FileNotFoundError as e:
            logger.error(f"Failed to initialize WhisperCppTranscriber: {e}")
            logger.error("Please ensure paths for whisper.cpp are correct (WHISPER_CPP_DIR, WHISPER_CLI_PATH, MODEL_PATH), "
                         "whisper.cpp is built (e.g., by running 'make' in WHISPER_CPP_DIR), and the model is downloaded to the 'models' subdirectory.")
            logger.error("You might need to set WHISPER_CPP_DIR, WHISPER_CLI_PATH, MODEL_PATH environment variables or edit their default values in main.py.")
            logger.error("Refer to whisper.cpp documentation and main_app.py for path configuration examples.")
            return 1
        except Exception as e:
            logger.error(f"An unexpected error occurred during transcriber initialization: {e}", exc_info=True)
            return 1
        logger.info("whisper.cpp transcriber initialized.")

        # Initialize Gemma analyzer
        analyzer = GemmaAnalyzer()

        # Check LMStudio connection
        lmstudio_available = analyzer.check_connection()
        if not lmstudio_available:
            logger.warning("\nWARNING: Could not connect to LMStudio API.")
            logger.info("Make sure LMStudio is running and the Gemma model is loaded.")
            logger.info("You should:")
            logger.info("1. Open LMStudio")
            logger.info("2. Load the Gemma model (e.g., qwen2-7b)")
            logger.info("3. Click on 'Local Server' and start the server")

            user_input = input("Continue without Gemma analysis? (y/n): ")
            if user_input.lower() != 'y':
                return 1

        # Initialize TTS system
        tts = SystemTTS(
            language="ru" if analyzer.system_prompt and "русский" in analyzer.system_prompt.lower() else "en"
        )

        # Main application loop
        response_files: List[str] = []  # Keep track of response files for cleanup

        while True:
            user_input = input(f"\nPress Enter to start recording for {RECORD_SECONDS}s (or type 'exit' to quit): ")
            if user_input.lower() == 'exit':
                break

            # Record audio from microphone
            audio_file = record_audio(
                record_seconds=RECORD_SECONDS,
                input_device_index=MICROPHONE_INDEX
            )
            if not audio_file: # record_audio might return None or raise error on failure
                logger.error("Audio recording failed. Skipping this iteration.")
                continue


            # Transcribe the recorded audio using whisper.cpp
            logger.info("Transcribing audio with whisper.cpp...")
            try:
                transcription = transcriber.get_transcription_text(audio_file)
            except RuntimeError as e:
                logger.error(f"Transcription failed: {e}")
                # Clean up the failed recording
                if os.path.exists(audio_file):
                    os.remove(audio_file)
                continue # Skip to next iteration
            except Exception as e:
                logger.error(f"An unexpected error occurred during transcription: {e}", exc_info=True)
                if os.path.exists(audio_file):
                    os.remove(audio_file)
                continue


            logger.info("\nTranscription:")
            logger.info(f"\"{transcription}\"")

            # Analyze with Gemma if LMStudio is available
            analysis_text_to_speak: Optional[str] = None
            if lmstudio_available:
                try:
                    logger.info("\nAnalyzing with Gemma...")
                    analysis_result = analyzer.analyze_text(transcription)
                    logger.info("\nGemma Analysis:")
                    logger.info(analysis_result)
                    analysis_text_to_speak = analysis_result
                except Exception as e:
                    logger.error(f"\nFailed to analyze with Gemma: {str(e)}")
                    analysis_text_to_speak = f"I couldn't analyze that with Gemma. I heard you say: {transcription}"
            else:
                analysis_text_to_speak = f"Gemma analysis is not available. I heard you say: {transcription}"

            # Generate speech response
            if analysis_text_to_speak:
                logger.info("\nGenerating speech with System TTS...")
                # SystemTTS directly plays audio and returns a dummy file path for cleanup logic
                temp_speech_file = tts.generate_speech(analysis_text_to_speak)
                if temp_speech_file:
                    response_files.append(temp_speech_file)
                    # Playback is handled by tts.generate_speech for SystemTTS in its current form.
                    # If tts.generate_speech were to *only* create a file, then play_audio() would be needed here.
                    # logger.info("Playing response...")
                    # play_audio(temp_speech_file) # This might be redundant if SystemTTS plays directly
                else:
                    logger.error("Failed to generate speech.")


            # Clean up recording file
            if os.path.exists(audio_file):
                try:
                    os.remove(audio_file)
                    logger.info(f"Cleaned up temporary audio file: {audio_file}")
                except OSError as e:
                    logger.warning(f"Could not remove temporary audio file {audio_file}: {e}")


    except KeyboardInterrupt:
        logger.info("\nExiting program...")
        return 0
    except Exception as e:
        logger.error(f"An critical error occurred in the main loop: {e}")
        traceback.print_exc()
        return 1
    finally:
        # Clean up any generated dummy response files
        logger.info("Cleaning up temporary files...")
        clean_temp_files(response_files)
        # Clean up any leftover .json files in temp_recordings from whisper.cpp if main_app.py's cleanup isn't running
        # Note: transcription_cpp.py should clean its own .json files.
        # This is more of a safeguard if something went wrong or if other .json files accumulated.
        temp_dir = "temp_recordings"
        if os.path.exists(temp_dir):
            for item in os.listdir(temp_dir):
                if item.endswith(".json"):
                    try:
                        os.remove(os.path.join(temp_dir, item))
                        logger.info(f"Cleaned up stray JSON file: {os.path.join(temp_dir, item)}")
                    except OSError as e:
                        logger.warning(f"Could not remove stray JSON file {os.path.join(temp_dir, item)}: {e}")


    return 0


if __name__ == "__main__":
    sys.exit(main())