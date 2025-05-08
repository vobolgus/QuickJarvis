"""
Voice Assistant main application that integrates speech recognition,
text analysis, and speech synthesis components.
"""
import os
import sys
import traceback
from typing import List, Optional

# Import modules
from utils import logger, clean_temp_files, check_ffmpeg_installed
from recording import record_audio, DEFAULT_RECORD_SECONDS, list_audio_devices
from transcription_cpp import WhisperCppTranscriber
from analysis import GemmaAnalyzer
# playback.py is not directly used in main loop if SystemTTS plays directly
# from playback import play_audio
from tts import SystemTTS

# --- Configuration for whisper.cpp ---
DEFAULT_WHISPER_CPP_DIR = os.path.expanduser("~/dev/whisper.cpp")
WHISPER_CPP_DIR = os.getenv("WHISPER_CPP_DIR", DEFAULT_WHISPER_CPP_DIR)
DEFAULT_WHISPER_CLI_PATH = os.path.join(WHISPER_CPP_DIR, "build/bin/whisper-cli")
WHISPER_CLI_PATH = os.getenv("WHISPER_CLI_PATH", DEFAULT_WHISPER_CLI_PATH)
MODEL_FILENAME = "ggml-base.en.bin"
DEFAULT_MODEL_PATH = os.path.join(WHISPER_CPP_DIR, "models", MODEL_FILENAME)
MODEL_PATH = os.getenv("WHISPER_MODEL_PATH", DEFAULT_MODEL_PATH)
RECORD_SECONDS = DEFAULT_RECORD_SECONDS
MICROPHONE_INDEX = None
# --- End Configuration ---


def main() -> int:
    """
    Main function that runs the Voice Assistant application.

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    print("\n🎤 Voice Assistant Activated 🎤")
    logger.debug(f"Recording duration: {RECORD_SECONDS} seconds")
    logger.debug(f"Using WHISPER_CPP_DIR: {WHISPER_CPP_DIR}")
    logger.debug(f"Using WHISPER_CLI_PATH: {WHISPER_CLI_PATH}")
    logger.debug(f"Using MODEL_PATH: {MODEL_PATH}")

    if "--list-devices" in sys.argv:
        list_audio_devices()
        return 0

    if not check_ffmpeg_installed():
        return 1

    try:
        logger.debug("Initializing whisper.cpp transcriber...")
        try:
            transcriber = WhisperCppTranscriber(
                model_path=MODEL_PATH,
                whisper_cli_path=WHISPER_CLI_PATH
            )
        except FileNotFoundError as e:
            logger.error(f"Failed to initialize WhisperCppTranscriber: {e}")
            print("ERROR: Whisper C++ transcriber setup failed. Please check paths and build.")
            print("Ensure WHISPER_CPP_DIR, WHISPER_CLI_PATH, MODEL_PATH are correct,")
            print("whisper.cpp is built (e.g., 'make' in WHISPER_CPP_DIR), and model is downloaded.")
            return 1
        except Exception as e:
            logger.error(f"An unexpected error occurred during transcriber initialization: {e}", exc_info=True)
            print("ERROR: An unexpected error occurred while initializing the transcriber.")
            return 1
        logger.debug("whisper.cpp transcriber initialized.")

        analyzer = GemmaAnalyzer()
        lmstudio_available = analyzer.check_connection()
        if not lmstudio_available:
            logger.warning("Could not connect to LMStudio API.")
            print("\n⚠️ WARNING: Could not connect to LMStudio API.")
            print("Please ensure LMStudio is running and the Gemma model is loaded:")
            print("1. Open LMStudio")
            print("2. Load the Gemma model (e.g., qwen2-7b)")
            print("3. Click on 'Local Server' and start the server")

            user_input = input("Continue without Gemma analysis? (y/n): ")
            if user_input.lower() != 'y':
                return 1

        tts = SystemTTS(
            language="ru" if analyzer.system_prompt and "русский" in analyzer.system_prompt.lower() else "en"
        )

        response_files: List[str] = []

        while True:
            user_input_prompt = f"\nPress Enter to start recording ({RECORD_SECONDS}s) or type 'exit': "
            user_cmd = input(user_input_prompt)
            if user_cmd.lower() == 'exit':
                break

            audio_file = record_audio(
                record_seconds=RECORD_SECONDS,
                input_device_index=MICROPHONE_INDEX
            )
            if not audio_file:
                logger.error("Audio recording failed. Skipping this iteration.")
                print("Audio recording failed. Please try again.")
                continue

            logger.debug("Transcribing audio with whisper.cpp...")
            try:
                transcription = transcriber.get_transcription_text(audio_file)
            except RuntimeError as e:
                logger.error(f"Transcription failed: {e}")
                print("Sorry, I couldn't transcribe that. Please try again.")
                if os.path.exists(audio_file): os.remove(audio_file)
                continue
            except Exception as e:
                logger.error(f"An unexpected error occurred during transcription: {e}", exc_info=True)
                print("An unexpected error occurred during transcription. Please try again.")
                if os.path.exists(audio_file): os.remove(audio_file)
                continue

            print(f"\nYOU: \"{transcription}\"")

            assistant_response_text: Optional[str] = None
            if lmstudio_available:
                try:
                    logger.debug("Analyzing with Gemma...")
                    analysis_result = analyzer.analyze_text(transcription)
                    assistant_response_text = analysis_result
                except Exception as e:
                    logger.error(f"\nFailed to analyze with Gemma: {str(e)}")
                    assistant_response_text = f"I couldn't analyze that with Gemma. I heard you say: \"{transcription}\""
            else:
                assistant_response_text = f"Gemma analysis is not available. I heard you say: \"{transcription}\""

            if assistant_response_text:
                print(f"ASSISTANT: {assistant_response_text}")
                logger.debug("Generating speech with System TTS...")
                temp_speech_file = tts.generate_speech(assistant_response_text)
                if temp_speech_file:
                    response_files.append(temp_speech_file)
                else:
                    logger.error("Failed to generate speech.")
                    print("Sorry, I encountered an issue with speech synthesis.")

            if os.path.exists(audio_file):
                try:
                    os.remove(audio_file)
                    logger.debug(f"Cleaned up temporary audio file: {audio_file}")
                except OSError as e:
                    logger.warning(f"Could not remove temporary audio file {audio_file}: {e}")

    except KeyboardInterrupt:
        print("\nExiting program...")
        return 0
    except Exception as e:
        logger.error(f"An critical error occurred in the main loop: {e}")
        traceback.print_exc() # Print full traceback for critical errors
        print(f"A critical error occurred: {e}")
        return 1
    finally:
        logger.debug("Cleaning up temporary files...")
        clean_temp_files(response_files)
        temp_dir = "temp_recordings"
        if os.path.exists(temp_dir):
            for item in os.listdir(temp_dir):
                if item.endswith(".json"):
                    try:
                        os.remove(os.path.join(temp_dir, item))
                        logger.debug(f"Cleaned up stray JSON file: {os.path.join(temp_dir, item)}")
                    except OSError as e:
                        logger.warning(f"Could not remove stray JSON file {os.path.join(temp_dir, item)}: {e}")
    return 0

if __name__ == "__main__":
    sys.exit(main())