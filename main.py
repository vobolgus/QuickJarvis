"""
Voice Assistant main application that integrates speech recognition,
text analysis, and speech synthesis components with voice cloning support.
Includes wake word detection and conversation support.
"""
import os
import sys
import traceback
import time
import argparse
from typing import List, Optional, Dict, Any

# Import modules
from utils import logger, clean_temp_files, check_ffmpeg_installed, get_timestamp
from recording import record_audio, DEFAULT_RECORD_SECONDS, list_audio_devices
from transcription_cpp import WhisperCppTranscriber
from analysis import GemmaAnalyzer
from tts import TTSManager, SystemTTS, VoiceCloningTTS
from playback import play_audio

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
FOLLOW_UP_LISTEN_SECONDS = 10 # How long to listen for a follow-up after assistant speaks (VAD timeout)
# --- End Follow-up Configuration ---

# --- Gemma Exit Phrases Configuration ---
# Phrases that, if found in Gemma's response (case-insensitive), will trigger a shutdown.
# Align these with the instructions in GemmaAnalyzer's DEFAULT_SYSTEM_PROMPT.
GEMMA_EXIT_PHRASES = ["goodbye", "session ended", "farewell", "ending conversation", "terminating session"]
# --- End Gemma Exit Phrases Configuration ---

# --- Voice Cloning Configuration ---
DEFAULT_TTS_TYPE = "system"  # "system" or "cloned"
DEFAULT_VOICE = None  # Default system voice or cloned voice name
DEFAULT_LANGUAGE = "en"  # Default language
# --- End Voice Cloning Configuration ---


def play_response(response_file_path):
    """Helper to play a response audio file immediately."""
    if response_file_path:
        try:
            success = play_audio(response_file_path)
            if not success:
                logger.warning(f"Failed to play audio file: {response_file_path}")
                print("[Audio playback failed - check your audio settings]")
            return response_file_path
        except Exception as e:
            logger.error(f"Error during audio playback: {e}")
            print("[Audio playback error - see logs for details]")
    return response_file_path


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Voice Assistant with Voice Cloning")

    # General options
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices and exit")
    parser.add_argument("--mic-index", type=int, help="Microphone device index to use")

    # Wake word options
    parser.add_argument("--wake-word", type=str, default=WAKE_WORD, help="Wake word to activate the assistant")
    parser.add_argument("--wake-activation", type=str, default=WAKE_WORD_ACTIVATION_SOUND,
                        help="Sound or phrase spoken upon wake word detection")

    # TTS options
    tts_group = parser.add_argument_group("Text-to-Speech Options")
    tts_group.add_argument("--tts-type", choices=["system", "cloned"], default=DEFAULT_TTS_TYPE,
                         help="Type of TTS to use (system or cloned)")
    tts_group.add_argument("--voice", type=str, help="Voice to use (system voice name or cloned voice name)")
    tts_group.add_argument("--language", type=str, default=DEFAULT_LANGUAGE,
                         help="Language code for TTS (e.g., 'en', 'fr', 'es')")

    # Voice cloning options
    vc_group = parser.add_argument_group("Voice Cloning Options")
    vc_group.add_argument("--list-cloned-voices", action="store_true",
                        help="List available cloned voices and exit")
    vc_group.add_argument("--add-voice", action="store_true",
                        help="Add a new cloned voice")
    vc_group.add_argument("--voice-sample", type=str,
                        help="Path to voice sample file for adding a new voice")
    vc_group.add_argument("--voice-name", type=str,
                        help="Name for the new voice when adding")
    vc_group.add_argument("--remove-voice", type=str,
                        help="Name of cloned voice to remove")

    return parser.parse_args()


def add_new_voice(tts_manager, voice_sample, voice_name):
    """
    Add a new cloned voice to the system.

    Args:
        tts_manager: TTSManager instance
        voice_sample: Path to voice sample file
        voice_name: Name for the new voice

    Returns:
        True if voice was added successfully, False otherwise
    """
    print(f"\n🎤 Adding new voice '{voice_name}' from sample: {voice_sample}")

    if not os.path.exists(voice_sample):
        print(f"❌ Voice sample file not found: {voice_sample}")
        return False

    # Add basic metadata
    metadata = {
        "Created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Source": os.path.basename(voice_sample)
    }

    # Add the voice
    result = tts_manager.add_cloned_voice(voice_sample, voice_name, metadata)

    if result:
        print(f"✅ Voice '{voice_name}' added successfully!")
        return True
    else:
        print(f"❌ Failed to add voice '{voice_name}'")
        return False


def list_cloned_voices(tts_manager):
    """
    List available cloned voices.

    Args:
        tts_manager: TTSManager instance
    """
    voices = tts_manager.list_available_voices(tts_type="cloned")

    print("\n🎤 Available Cloned Voices:")
    if not voices:
        print("  No cloned voices available")
        return

    for voice in voices:
        name = voice["name"]
        has_sample = "✓" if voice.get("has_sample", False) else "✗"
        has_embedding = "✓" if voice.get("has_embedding", False) else "✗"
        metadata = voice.get("metadata", {})

        print(f"  • {name}")
        print(f"    - Sample: {has_sample}")
        print(f"    - Embedding: {has_embedding}")

        if metadata:
            print("    - Metadata:")
            for key, value in metadata.items():
                print(f"      • {key}: {value}")
        print()


def remove_cloned_voice(tts_manager, voice_name):
    """
    Remove a cloned voice from the system.

    Args:
        tts_manager: TTSManager instance
        voice_name: Name of the voice to remove

    Returns:
        True if voice was removed successfully, False otherwise
    """
    print(f"\n🗑️ Removing cloned voice: {voice_name}")

    result = tts_manager.remove_cloned_voice(voice_name)

    if result:
        print(f"✅ Voice '{voice_name}' removed successfully!")
        return True
    else:
        print(f"❌ Failed to remove voice '{voice_name}'")
        return False


def main():
    """
    Main function that runs the Voice Assistant application with wake word detection
    and voice cloning support.
    """
    args = parse_arguments()

    # Update global variables from args
    global WAKE_WORD, WAKE_WORD_ACTIVATION_SOUND, MICROPHONE_INDEX
    WAKE_WORD = args.wake_word
    WAKE_WORD_ACTIVATION_SOUND = args.wake_activation
    MICROPHONE_INDEX = args.mic_index

    # Initialize TTS manager with CLI args
    tts_manager = TTSManager(
        default_type=args.tts_type,
        default_voice=args.voice,
        default_language=args.language
    )

    # Handle special commands
    if args.list_devices:
        list_audio_devices()
        return 0

    if args.list_cloned_voices:
        list_cloned_voices(tts_manager)
        return 0

    if args.add_voice:
        if not args.voice_sample or not args.voice_name:
            print("❌ Error: --voice-sample and --voice-name are required when using --add-voice")
            return 1
        success = add_new_voice(tts_manager, args.voice_sample, args.voice_name)
        return 0 if success else 1

    if args.remove_voice:
        success = remove_cloned_voice(tts_manager, args.remove_voice)
        return 0 if success else 1

    print(f"\n🎤 Voice Assistant Activated. Wake word: '{WAKE_WORD.upper()}' 🎤")
    if args.tts_type == "cloned" and args.voice:
        print(f"📢 Using cloned voice: '{args.voice}'")
    logger.debug(f"Using WHISPER_CPP_DIR: {WHISPER_CPP_DIR}")
    logger.debug(f"Using WHISPER_CLI_PATH: {WHISPER_CLI_PATH}")
    logger.debug(f"Using MODEL_PATH: {MODEL_PATH}")
    logger.info(f"Wake word detection will use Voice Activity Detection (VAD) with a max duration of {WAKE_WORD_RECORD_SECONDS}s.")
    logger.info("Transcription of detected speech for wake word is still CPU-intensive. A dedicated wake word engine is recommended for real-world use.")
    logger.info(f"Command recording will use Voice Activity Detection (VAD) with a max duration of {COMMAND_MAX_DURATION_SECONDS}s.")
    logger.info(f"Follow-up listening will use VAD with a max duration of {FOLLOW_UP_LISTEN_SECONDS}s.")

    if not check_ffmpeg_installed():
        return 1

    response_files = []  # Keep track of response files for cleanup

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

        # Main application loop: Listen for wake word, then command
        while True:
            print(f"\n👂 Listening for wake word '{WAKE_WORD}'...")

            # Record a short audio chunk for wake word detection (with VAD)
            audio_chunk_file = record_audio(
                record_seconds=WAKE_WORD_RECORD_SECONDS, # Acts as VAD timeout for wake word
                input_device_index=MICROPHONE_INDEX,
                suppress_prints=True, # Suppress "Recording..." messages for wake word chunks
                use_vad=True # Use VAD for wake word detection
            )

            if not audio_chunk_file:
                # This means VAD timed out without detecting speech, or recording failed
                continue

            transcription_chunk = ""
            try:
                # Transcribe the chunk
                transcription_chunk = transcriber.get_transcription_text(audio_chunk_file).lower()
                logger.debug(f"Wake word check: Transcribed chunk: \"{transcription_chunk}\"")
            except Exception as e:
                logger.error(f"Error during wake word chunk transcription: {e}")
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
                    ack_speech_file = tts_manager.generate_speech(WAKE_WORD_ACTIVATION_SOUND)
                    if ack_speech_file:
                        response_files.append(ack_speech_file)
                        # Play the activation sound
                        play_response(ack_speech_file)

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
                    continue # Back to wake word listening

                current_transcribed_text = ""
                try:
                    logger.debug("Transcribing initial command with whisper.cpp...")
                    current_transcribed_text = transcriber.get_transcription_text(command_audio_file).strip()
                except Exception as e:
                    logger.error(f"Error during initial command transcription: {e}")
                    print("Sorry, I couldn't transcribe your command. Please try again after the wake word.")
                finally:
                    if os.path.exists(command_audio_file):
                        try:
                            os.remove(command_audio_file)
                            logger.debug(f"Cleaned up initial command audio file: {command_audio_file}")
                        except OSError as e_os:
                            logger.warning(f"Could not remove initial command audio file {command_audio_file}: {e_os}")

                if not current_transcribed_text:
                    logger.info("Transcription resulted in empty text.")
                    print("I didn't catch that. Let's try again.")
                    continue # Back to wake word listening

                # --- Conversation Loop (handles initial command and subsequent follow-ups) ---
                while True: # Inner conversation loop
                    print(f"\nYOU: \"{current_transcribed_text}\"")

                    # Check for exit command
                    if current_transcribed_text.strip().lower() == "exit":
                        print("Exit command received. Shutting down...")
                        farewell_speech_file = tts_manager.generate_speech("Goodbye!")
                        if farewell_speech_file:
                            response_files.append(farewell_speech_file)
                            # Play the farewell message
                            play_response(farewell_speech_file)
                        return 0 # Exit main function successfully

                    # Check for voice cloning commands in the transcription
                    if "use system voice" in current_transcribed_text.lower():
                        print("Switching to system TTS...")
                        tts_manager.default_type = "system"
                        assistant_response_text = "I've switched to using the system voice."
                        print(f"ASSISTANT: {assistant_response_text}")
                        temp_speech_file = tts_manager.generate_speech(assistant_response_text)
                        if temp_speech_file:
                            response_files.append(temp_speech_file)
                            # Play the response
                            play_response(temp_speech_file)
                    elif "use cloned voice" in current_transcribed_text.lower():
                        # Extract voice name from command
                        parts = current_transcribed_text.lower().split("use cloned voice")
                        if len(parts) > 1 and parts[1].strip():
                            voice_name = parts[1].strip()
                            # Check if voice exists
                            voices = tts_manager.list_available_voices(tts_type="cloned")
                            voice_names = [v["name"] for v in voices]

                            if voice_name in voice_names:
                                print(f"Switching to cloned voice: '{voice_name}'")
                                tts_manager.default_type = "cloned"
                                tts_manager.default_voice = voice_name
                                assistant_response_text = f"I've switched to using the cloned voice named {voice_name}."
                            else:
                                assistant_response_text = f"I couldn't find a cloned voice named {voice_name}. Available voices are: {', '.join(voice_names) if voice_names else 'None'}"
                        else:
                            voices = tts_manager.list_available_voices(tts_type="cloned")
                            voice_names = [v["name"] for v in voices]
                            if voice_names:
                                assistant_response_text = f"Please specify which cloned voice to use. Available voices are: {', '.join(voice_names)}"
                            else:
                                assistant_response_text = "There are no cloned voices available. Please add a voice first."

                        print(f"ASSISTANT: {assistant_response_text}")
                        temp_speech_file = tts_manager.generate_speech(assistant_response_text)
                        if temp_speech_file:
                            response_files.append(temp_speech_file)
                            # Play the response
                            play_response(temp_speech_file)
                    elif "clone my voice" in current_transcribed_text.lower() or "add my voice" in current_transcribed_text.lower():
                        print("Starting voice cloning process...")
                        assistant_response_text = "I'll clone your voice. Please speak for 20 seconds after the beep."
                        print(f"ASSISTANT: {assistant_response_text}")
                        temp_speech_file = tts_manager.generate_speech(assistant_response_text)
                        if temp_speech_file:
                            response_files.append(temp_speech_file)
                            # Play the response
                            play_response(temp_speech_file)

                        # Record a longer sample for voice cloning
                        time.sleep(1)  # Short pause before recording
                        print("🔴 Recording voice sample (20 seconds)... Speak naturally.")
                        voice_sample_file = record_audio(
                            record_seconds=20,
                            input_device_index=MICROPHONE_INDEX,
                            suppress_prints=False,
                            use_vad=False  # We want a fixed duration for the sample
                        )

                        if not voice_sample_file:
                            print("❌ Failed to record voice sample.")
                            assistant_response_text = "I couldn't record your voice sample. Please try again."
                        else:
                            # Extract a name for the voice
                            print("Processing voice sample...")
                            voice_name = f"user_voice_{get_timestamp()}"

                            # Add the voice
                            result = tts_manager.add_cloned_voice(voice_sample_file, voice_name, {"Source": "User recording"})

                            if result:
                                tts_manager.default_type = "cloned"
                                tts_manager.default_voice = voice_name
                                assistant_response_text = f"I've successfully cloned your voice! I'll use it from now on."
                            else:
                                assistant_response_text = "I had trouble processing your voice sample. Please try again."

                        print(f"ASSISTANT: {assistant_response_text}")
                        temp_speech_file = tts_manager.generate_speech(assistant_response_text)
                        if temp_speech_file:
                            response_files.append(temp_speech_file)
                            # Play the response
                            play_response(temp_speech_file)
                    elif "list voices" in current_transcribed_text.lower():
                        voices = tts_manager.list_available_voices(tts_type="cloned")
                        voice_names = [v["name"] for v in voices]

                        if voice_names:
                            assistant_response_text = f"Available cloned voices are: {', '.join(voice_names)}"
                        else:
                            assistant_response_text = "There are no cloned voices available. Please add a voice first."

                        print(f"ASSISTANT: {assistant_response_text}")
                        temp_speech_file = tts_manager.generate_speech(assistant_response_text)
                        if temp_speech_file:
                            response_files.append(temp_speech_file)
                            # Play the response
                            play_response(temp_speech_file)
                    else:
                        # Process normal command with Gemma
                        if lmstudio_available:
                            logger.debug("Analyzing command with Gemma...")
                            assistant_response_text = analyzer.analyze_text(current_transcribed_text)
                        else:
                            assistant_response_text = f"Gemma analysis is not available. I heard you say: \"{current_transcribed_text}\""

                        if assistant_response_text is None:
                            logger.error("GemmaAnalyzer.analyze_text unexpectedly returned None.")
                            assistant_response_text = "I encountered an internal error trying to understand that."

                        print(f"ASSISTANT: {assistant_response_text}")
                        logger.debug("Generating speech response with TTS Manager...")
                        temp_speech_file = tts_manager.generate_speech(assistant_response_text)
                        if temp_speech_file:
                            response_files.append(temp_speech_file)
                            # Play the response
                            play_response(temp_speech_file)
                        else:
                            logger.error("Failed to generate speech response.")

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
                    except Exception as e:
                        logger.error(f"Error during follow-up transcription: {e}")
                        print("Sorry, I couldn't transcribe your follow-up. Returning to wake word listening.")
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