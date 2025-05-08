"""
Voice Assistant main application that integrates speech recognition,
text analysis, and speech synthesis components.
Includes a simulated wake word detection.
"""
import os
import sys
import traceback
import time
from typing import List, Optional

# Import modules
from utils import logger, clean_temp_files, check_ffmpeg_installed
from recording import record_audio, DEFAULT_RECORD_SECONDS, list_audio_devices
from transcription_cpp import WhisperCppTranscriber
from analysis import GemmaAnalyzer
from tts import SystemTTS

# --- Configuration for whisper.cpp ---
DEFAULT_WHISPER_CPP_DIR = os.path.expanduser("~/dev/whisper.cpp")
WHISPER_CPP_DIR = os.getenv("WHISPER_CPP_DIR", DEFAULT_WHISPER_CPP_DIR)
DEFAULT_WHISPER_CLI_PATH = os.path.join(WHISPER_CPP_DIR, "build/bin/whisper-cli")
WHISPER_CLI_PATH = os.getenv("WHISPER_CLI_PATH", DEFAULT_WHISPER_CLI_PATH)
MODEL_FILENAME = "ggml-base.en.bin"
DEFAULT_MODEL_PATH = os.path.join(WHISPER_CPP_DIR, "models", MODEL_FILENAME)
MODEL_PATH = os.getenv("WHISPER_MODEL_PATH", DEFAULT_MODEL_PATH)
MICROPHONE_INDEX = None
# --- End Configuration ---

# --- Wake Word Configuration ---
WAKE_WORD = "computer"  # Change this to your desired wake word
# How long to record audio chunks for wake word detection (simulated - will be slow)
WAKE_WORD_RECORD_SECONDS = 3
# How long to record for the actual command after wake word detection (acts as a timeout for VAD)
COMMAND_MAX_DURATION_SECONDS = 10 # Max duration for command recording with VAD
WAKE_WORD_ACTIVATION_SOUND = "Yes?" # Sound/phrase spoken by TTS upon wake word detection
# --- End Wake Word Configuration ---

# --- Gemma Exit Phrases Configuration ---
# Phrases that, if found in Gemma's response (case-insensitive), will trigger a shutdown.
# Align these with the instructions in GemmaAnalyzer's DEFAULT_SYSTEM_PROMPT.
GEMMA_EXIT_PHRASES = ["goodbye", "session ended", "farewell", "ending conversation", "terminating session"]
# --- End Gemma Exit Phrases Configuration ---


def main() -> int:
    """
    Main function that runs the Voice Assistant application with wake word detection.
    """
    print(f"\n🎤 Voice Assistant Activated. Wake word: '{WAKE_WORD.upper()}' 🎤")
    logger.debug(f"Using WHISPER_CPP_DIR: {WHISPER_CPP_DIR}")
    logger.debug(f"Using WHISPER_CLI_PATH: {WHISPER_CLI_PATH}")
    logger.debug(f"Using MODEL_PATH: {MODEL_PATH}")
    logger.info(f"IMPORTANT: Wake word detection is SIMULATED by transcribing short audio chunks ({WAKE_WORD_RECORD_SECONDS}s).")
    logger.info("This will be slow and CPU-intensive. A dedicated wake word engine is recommended for real-world use.")
    logger.info(f"Command recording will use Voice Activity Detection (VAD) with a max duration of {COMMAND_MAX_DURATION_SECONDS}s.")


    if "--list-devices" in sys.argv:
        list_audio_devices()
        return 0

    if not check_ffmpeg_installed():
        return 1

    response_files: List[str] = []  # Keep track of response files for cleanup

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
            print("Gemma analysis will be unavailable. Ensure LMStudio is running with a model loaded.")
            # No longer asking to quit, will proceed without Gemma if unavailable

        tts = SystemTTS(
            language="ru" if analyzer.system_prompt and "русский" in analyzer.system_prompt.lower() else "en"
        )

        # Main application loop: Listen for wake word, then command
        while True:
            print(f"\n👂 Listening for wake word '{WAKE_WORD}'...")

            # Record a short audio chunk for wake word detection (fixed duration)
            audio_chunk_file = record_audio(
                record_seconds=WAKE_WORD_RECORD_SECONDS,
                input_device_index=MICROPHONE_INDEX,
                suppress_prints=True, # Suppress "Recording..." messages for wake word chunks
                use_vad=False # Fixed duration for wake word
            )

            if not audio_chunk_file:
                logger.warning("Wake word recording chunk failed. Retrying in 1s.")
                time.sleep(1) # Avoid busy-looping on persistent recording errors
                continue

            transcription_chunk = ""
            try:
                # Transcribe the chunk
                transcription_chunk = transcriber.get_transcription_text(audio_chunk_file).lower()
                logger.debug(f"Wake word check: Transcribed chunk: \"{transcription_chunk}\"")
            except RuntimeError as e:
                logger.error(f"Transcription of wake word chunk failed: {e}")
                # Don't stop the loop, just try again
            except Exception as e:
                logger.error(f"Unexpected error during wake word chunk transcription: {e}", exc_info=True)
            finally:
                if os.path.exists(audio_chunk_file):
                    try:
                        os.remove(audio_chunk_file)
                    except OSError as e:
                        logger.warning(f"Could not remove wake word audio chunk {audio_chunk_file}: {e}")

            # Check if the wake word is in the transcribed chunk
            if WAKE_WORD.lower() in transcription_chunk:
                print(f"✨ Wake word '{WAKE_WORD.upper()}' detected!")
                if WAKE_WORD_ACTIVATION_SOUND:
                    ack_speech_file = tts.generate_speech(WAKE_WORD_ACTIVATION_SOUND)
                    if ack_speech_file:
                        response_files.append(ack_speech_file)

                # For command recording, use VAD.
                # record_seconds here acts as the max_duration/timeout for VAD.
                command_audio_file = record_audio(
                    record_seconds=COMMAND_MAX_DURATION_SECONDS,
                    input_device_index=MICROPHONE_INDEX,
                    suppress_prints=False, # Show VAD messages like "Listening for speech..."
                    use_vad=True
                )

                if not command_audio_file:
                    logger.error("Command audio recording failed or no speech detected.")
                    # Message already printed by record_audio if VAD fails due to no speech
                    # print("Sorry, I couldn't record your command (no speech detected or error). Listening for wake word again.")
                    continue # Go back to wake word listening

                transcription_command = ""
                try:
                    logger.debug("Transcribing command with whisper.cpp...")
                    transcription_command = transcriber.get_transcription_text(command_audio_file)

                    if not transcription_command.strip():
                        logger.info("Transcription resulted in empty text. Assuming no command was given.")
                        print("You didn't say anything? Listening for wake word again.")
                        if os.path.exists(command_audio_file): os.remove(command_audio_file)
                        continue

                    print(f"\nYOU: \"{transcription_command}\"")

                    if transcription_command.strip().lower() == "exit":
                        print("Exit command received. Shutting down...")
                        # Generate a specific "Goodbye!" for direct exit command
                        farewell_speech_file = tts.generate_speech("Goodbye!")
                        if farewell_speech_file:
                             response_files.append(farewell_speech_file) # Add for cleanup, though app exits soon
                        # Clean up current command audio before exiting
                        if os.path.exists(command_audio_file): os.remove(command_audio_file)
                        return 0 # Exit main function successfully

                    assistant_response_text: Optional[str] = None
                    if lmstudio_available:
                        try:
                            logger.debug("Analyzing command with Gemma...")
                            analysis_result = analyzer.analyze_text(transcription_command)
                            assistant_response_text = analysis_result
                        except Exception as e:
                            logger.error(f"Failed to analyze command with Gemma: {str(e)}")
                            assistant_response_text = f"I couldn't analyze that with Gemma. I heard you say: \"{transcription_command}\""
                    else:
                        assistant_response_text = f"Gemma analysis is not available. I heard you say: \"{transcription_command}\""

                    if assistant_response_text:
                        print(f"ASSISTANT: {assistant_response_text}")
                        logger.debug("Generating speech response with System TTS...")
                        temp_speech_file = tts.generate_speech(assistant_response_text)
                        if temp_speech_file:
                            response_files.append(temp_speech_file)
                        else:
                            logger.error("Failed to generate speech response.")
                            print("Sorry, I encountered an issue with speech synthesis.")

                        # Check if Gemma's response indicates an intent to exit
                        normalized_gemma_response = assistant_response_text.lower()
                        for phrase in GEMMA_EXIT_PHRASES:
                            if phrase in normalized_gemma_response:
                                print(f"Gemma indicated session end with: '{assistant_response_text}'. Shutting down...")
                                logger.info(f"Exiting based on Gemma's response containing exit cue: '{phrase}'")
                                # The assistant_response_text (which contains the goodbye) has already been spoken.
                                if os.path.exists(command_audio_file): os.remove(command_audio_file)
                                return 0 # Exit main function successfully


                except RuntimeError as e:
                    logger.error(f"Transcription of command failed: {e}")
                    print("Sorry, I couldn't transcribe your command. Please try again after the wake word.")
                except Exception as e:
                    logger.error(f"An unexpected error occurred during command processing: {e}", exc_info=True)
                    print("An unexpected error occurred. Listening for wake word again.")
                finally:
                    if os.path.exists(command_audio_file):
                        try:
                            os.remove(command_audio_file)
                            logger.debug(f"Cleaned up command audio file: {command_audio_file}")
                        except OSError as e:
                            logger.warning(f"Could not remove command audio file {command_audio_file}: {e}")
            # else: # Wake word not detected, loop continues.
                # time.sleep(0.1) # Optional small delay if wake word detection is very fast
                                # Not strictly necessary with slow Whisper.cpp simulation

    except KeyboardInterrupt:
        print("\nExiting program via KeyboardInterrupt...")
        return 0
    except Exception as e:
        logger.error(f"A critical error occurred in the main loop: {e}")
        traceback.print_exc()
        print(f"A critical error occurred: {e}")
        return 1
    finally:
        logger.debug("Cleaning up temporary files...")
        clean_temp_files(response_files)
        temp_dir = "temp_recordings"
        if os.path.exists(temp_dir):
            for item in os.listdir(temp_dir):
                if item.endswith(".json"): # whisper.cpp JSON output files
                    try:
                        os.remove(os.path.join(temp_dir, item))
                        logger.debug(f"Cleaned up stray JSON file: {os.path.join(temp_dir, item)}")
                    except OSError as e:
                        logger.warning(f"Could not remove stray JSON file {os.path.join(temp_dir, item)}: {e}")
    return 0

if __name__ == "__main__":
    sys.exit(main())