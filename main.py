"""
Voice Assistant main application that integrates speech recognition,
text analysis, and speech synthesis components.
"""
import os
import sys
import traceback
from typing import List, Optional

# Import modules
from utils import logger, clean_temp_files, check_ffmpeg_installed, get_device
from recording import record_audio
from transcription import WhisperTranscriber
from analysis import GemmaAnalyzer
from playback import play_audio
from tts import SystemTTS


def main() -> int:
    """
    Main function that runs the Voice Assistant application.

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    logger.info("\nVoice Assistant with Whisper, Gemma and Bark")
    logger.info("-------------------------------------------")
    logger.info("This program will:")
    logger.info("1. Record your voice and transcribe it with Whisper")
    logger.info("2. Analyze the transcription with Gemma")
    logger.info("3. Read the analysis back to you with Bark")
    logger.info("Recording duration: 5 seconds")

    # Check prerequisites
    if not check_ffmpeg_installed():
        return 1

    # Get device configuration
    device, _ = get_device()

    # Initialize components
    try:
        # Initialize Whisper model for transcription
        transcriber = WhisperTranscriber(device=device)

        # Initialize Gemma analyzer
        analyzer = GemmaAnalyzer()

        # Check LMStudio connection
        lmstudio_available = analyzer.check_connection()
        if not lmstudio_available:
            logger.warning("\nWARNING: Could not connect to LMStudio API.")
            logger.info("Make sure LMStudio is running and the Gemma model is loaded.")
            logger.info("You should:")
            logger.info("1. Open LMStudio")
            logger.info("2. Load the Gemma model")
            logger.info("3. Click on 'Local Server' and start the server")

            user_input = input("Continue without Gemma analysis? (y/n): ")
            if user_input.lower() != 'y':
                return 1

        # Initialize TTS system with auto-fallback
        tts = SystemTTS(
            language="ru" if analyzer.system_prompt and "русский" in analyzer.system_prompt.lower() else "en"
        )

        # Main application loop
        response_files: List[str] = []  # Keep track of response files for cleanup

        while True:
            user_input = input("\nPress Enter to start recording (or type 'exit' to quit): ")
            if user_input.lower() == 'exit':
                break

            # Record audio from microphone
            audio_file = record_audio()

            # Transcribe the recorded audio
            logger.info("Transcribing audio...")
            transcription = transcriber.get_transcription_text(audio_file)

            logger.info("\nTranscription:")
            logger.info(transcription)

            # Analyze with Gemma if LMStudio is available
            analysis = None
            if lmstudio_available:
                try:
                    logger.info("\nAnalyzing with Gemma...")
                    analysis = analyzer.analyze_text(transcription)
                    logger.info("\nGemma Analysis:")
                    logger.info(analysis)
                except Exception as e:
                    logger.error(f"\nFailed to analyze with Gemma: {str(e)}")
                    analysis = f"I couldn't analyze that with Gemma. I heard you say: {transcription}"
            else:
                analysis = f"Gemma analysis is not available. I heard you say: {transcription}"

            # Generate speech response with Bark
            if analysis:
                logger.info("\nGenerating speech with Bark...")
                response_file = tts.generate_speech(analysis)
                if response_file:
                    response_files.append(response_file)
                    logger.info("Playing response...")
                    play_audio(response_file)

            # Clean up recording file
            os.remove(audio_file)

    except KeyboardInterrupt:
        logger.info("\nExiting program...")
        return 0
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        traceback.print_exc()
        return 1
    finally:
        # Clean up response files
        logger.info("Cleaning up temporary files...")
        clean_temp_files(response_files)

    return 0


if __name__ == "__main__":
    sys.exit(main())