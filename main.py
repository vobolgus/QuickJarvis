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

# --- Follow-up Configuration ---
FOLLOW_UP_LISTEN_SECONDS = 5 # How long to listen for a follow-up after assistant speaks (VAD timeout)
# --- End Follow-up Configuration ---

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
    logger.info(f"Wake word detection will use Voice Activity Detection (VAD) with a max duration of {WAKE_WORD_RECORD_SECONDS}s.")
    logger.info("Transcription of detected speech for wake word is still CPU-intensive. A dedicated wake word engine is recommended for real-world use.")
    logger.info(f"Command recording will use Voice Activity Detection (VAD) with a max duration of {COMMAND_MAX_DURATION_SECONDS}s.")
    logger.info(f"Follow-up listening will use VAD with a max duration of {FOLLOW_UP_LISTEN_SECONDS}s.")


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
        while True: # Outer Wake Word Loop
            print(f"\n👂 Listening for wake word '{WAKE_WORD}'...")

            # Record a short audio chunk for wake word detection (fixed duration)
            audio_chunk_file = record_audio(
                record_seconds=WAKE_WORD_RECORD_SECONDS, # Acts as VAD timeout for wake word
                input_device_index=MICROPHONE_INDEX,
                suppress_prints=True, # Suppress "Recording..." messages for wake word chunks
                use_vad=True # Use VAD for wake word detection
            )

            if not audio_chunk_file:
                # This means VAD timed out without detecting speech, or recording failed.
                # No explicit log here as record_audio with VAD already prints "Listening timed out, no speech detected"
                # or logs an error if the recording itself failed.
                # A short sleep can prevent spamming if there's a persistent recording issue not caught by VAD.
                # time.sleep(0.1) # Optional small delay
                continue

            transcription_chunk = ""
            try:
                # Transcribe the chunk
                transcription_chunk = transcriber.get_transcription_text(audio_chunk_file).lower()
                logger.debug(f"Wake word check: Transcribed chunk: \"{transcription_chunk}\"")
            except RuntimeError as e:
                logger.error(f"Transcription of wake word chunk failed: {e}")
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
                        response_files.append(ack_speech_file) # Add TTS marker file

                # --- Start of Conversation Turn: Get Initial Command ---
                print(f"👂 Listening for your command (VAD, max {COMMAND_MAX_DURATION_SECONDS}s)...")
                command_audio_file = record_audio(
                    record_seconds=COMMAND_MAX_DURATION_SECONDS,
                    input_device_index=MICROPHONE_INDEX,
                    suppress_prints=False,
                    use_vad=True
                )

                if not command_audio_file:
                    logger.info("Initial command audio recording failed or no speech detected.")
                    # Message already printed by record_audio if VAD fails due to no speech
                    continue # Back to wake word listening

                current_transcribed_text = ""
                try:
                    logger.debug("Transcribing initial command with whisper.cpp...")
                    current_transcribed_text = transcriber.get_transcription_text(command_audio_file).strip()
                except RuntimeError as e:
                    logger.error(f"Transcription of initial command failed: {e}")
                    print("Sorry, I couldn't transcribe your command. Please try again after the wake word.")
                except Exception as e:
                    logger.error(f"Unexpected error during initial command transcription: {e}", exc_info=True)
                    print("An unexpected error occurred. Listening for wake word again.")
                finally:
                    if os.path.exists(command_audio_file):
                        try:
                            os.remove(command_audio_file)
                            logger.debug(f"Cleaned up initial command audio file: {command_audio_file}")
                        except OSError as e_os:
                            logger.warning(f"Could not remove initial command audio file {command_audio_file}: {e_os}")

                if not current_transcribed_text:
                    if command_audio_file: # Implies recording was successful but transcription empty
                        logger.info("Transcription of initial command resulted in empty text.")
                        print("You didn't say anything? Listening for wake word again.")
                    # If command_audio_file was None, message already printed by record_audio or handled above
                    continue # Back to wake word listening

                # --- Conversation Loop (handles initial command and subsequent follow-ups) ---
                while True:
                    print(f"\nYOU: \"{current_transcribed_text}\"")

                    if current_transcribed_text.strip().lower() == "exit":
                        print("Exit command received. Shutting down...")
                        farewell_speech_file = tts.generate_speech("Goodbye!")
                        if farewell_speech_file:
                             response_files.append(farewell_speech_file) # Add TTS marker file
                        return 0 # Exit main function successfully

                    assistant_response_text: str
                    if lmstudio_available:
                        logger.debug("Analyzing command with Gemma...")
                        assistant_response_text = analyzer.analyze_text(current_transcribed_text)
                    else:
                        assistant_response_text = f"Gemma analysis is not available. I heard you say: \"{current_transcribed_text}\""

                    if assistant_response_text is None: # Defensive check, analyze_text should always return str
                        logger.error("GemmaAnalyzer.analyze_text unexpectedly returned None. This should not happen.")
                        assistant_response_text = "I encountered an internal error trying to understand that."


                    print(f"ASSISTANT: {assistant_response_text}")
                    logger.debug("Generating speech response with System TTS...")
                    temp_speech_file = tts.generate_speech(assistant_response_text)
                    if temp_speech_file:
                        response_files.append(temp_speech_file) # Add TTS marker file
                    else:
                        logger.error("Failed to generate speech response.")
                        # Assistant_response_text was already printed to console.

                    # Check if Gemma's response indicates an intent to exit
                    normalized_gemma_response = assistant_response_text.lower()
                    for phrase in GEMMA_EXIT_PHRASES:
                        if phrase in normalized_gemma_response:
                            print(f"Gemma indicated session end with: '{assistant_response_text}'. Shutting down...")
                            logger.info(f"Exiting based on Gemma's response containing exit cue: '{phrase}'")
                            return 0 # Exit main function successfully

                    # --- Follow-up Listening ---
                    print(f"\n👂 Listening for follow-up (VAD, max {FOLLOW_UP_LISTEN_SECONDS}s)...")
                    follow_up_audio_file = record_audio(
                        record_seconds=FOLLOW_UP_LISTEN_SECONDS,
                        input_device_index=MICROPHONE_INDEX,
                        suppress_prints=True, # Quieter for follow-up
                        use_vad=True
                    )

                    if not follow_up_audio_file:
                        logger.info("No follow-up speech detected or recording failed.")
                        print("...No follow-up. Returning to wake word listening.")
                        break # Exit conversation loop, go back to outer wake word loop

                    follow_up_transcription = ""
                    try:
                        logger.debug("Transcribing follow-up command...")
                        follow_up_transcription = transcriber.get_transcription_text(follow_up_audio_file).strip()
                    except RuntimeError as e:
                        logger.error(f"Transcription of follow-up failed: {e}")
                        print("Sorry, I couldn't transcribe your follow-up. Returning to wake word listening.")
                        break # Exit conversation loop
                    except Exception as e:
                        logger.error(f"Unexpected error during follow-up transcription: {e}", exc_info=True)
                        print("An unexpected error occurred with the follow-up. Returning to wake word listening.")
                        break # Exit conversation loop
                    finally:
                        if os.path.exists(follow_up_audio_file):
                            try:
                                os.remove(follow_up_audio_file)
                                logger.debug(f"Cleaned up follow-up audio file: {follow_up_audio_file}")
                            except OSError as e_os:
                                logger.warning(f"Could not remove follow-up audio file {follow_up_audio_file}: {e_os}")

                    if not follow_up_transcription:
                        logger.info("Follow-up transcription is empty.")
                        print("...No clear follow-up. Returning to wake word listening.")
                        break # Exit conversation loop

                    # If we have a valid follow-up transcription, update current_transcribed_text
                    # and the conversation loop will continue with this new text.
                    current_transcribed_text = follow_up_transcription
                # --- End of Conversation Loop ---
                # If 'break' is hit in the conversation loop, execution comes here,
                # then the outer wake word loop ('while True') continues.
            # else: Wake word not detected, main loop continues
                # time.sleep(0.1) # Optional small delay

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
        clean_temp_files(response_files) # Cleans TTS marker files
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