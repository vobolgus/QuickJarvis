"""
Speech-to-text functionality using whisper.cpp for the Voice Assistant.
This version reads JSON output from a file created by whisper-cli.
"""
import os
import subprocess
import json
from typing import Dict, Any

from utils import logger

class WhisperCppTranscriber:
    """
    Handles speech-to-text transcription using whisper.cpp's command-line interface.
    Assumes whisper-cli with -oj flag writes JSON to a <audio_filename>.json file.
    """

    def __init__(self, model_path: str, whisper_cli_path: str):
        """
        Initialize the Whisper C++ transcription system.

        Args:
            model_path: Path to the .bin ggml model file (e.g., /path/to/whisper.cpp/models/ggml-base.en.bin)
            whisper_cli_path: Path to the whisper-cli executable (e.g., /path/to/whisper.cpp/build/bin/whisper-cli)
        """
        self.model_path = os.path.abspath(model_path)
        self.whisper_cli_path = os.path.abspath(whisper_cli_path)

        logger.debug(f"Initializing WhisperCppTranscriber with:")
        logger.debug(f"  CLI Path: {self.whisper_cli_path}")
        logger.debug(f"  Model Path: {self.model_path}")

        if not os.path.isfile(self.whisper_cli_path) or not os.access(self.whisper_cli_path, os.X_OK):
            raise FileNotFoundError(
                f"whisper-cli not found or not executable at {self.whisper_cli_path}. "
                "Please check the path and ensure whisper.cpp is built."
            )
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"Whisper model not found at {self.model_path}. "
                "Please check the path and ensure the model is downloaded."
            )

        logger.debug("WhisperCppTranscriber initialized successfully.")

    def transcribe(self, audio_file: str) -> Dict[str, Any]:
        """
        Transcribe the given audio file to text using whisper-cli.
        Reads the JSON output from a file named <audio_file>.json.

        Args:
            audio_file: Path to the audio file to transcribe

        Returns:
            Dictionary containing the transcription result with at least a "text" key
            and "raw_whisper_cpp_output" holding the full JSON from whisper-cli.
        """
        abs_audio_file = os.path.abspath(audio_file)
        if not os.path.isfile(abs_audio_file):
            raise FileNotFoundError(f"Audio file not found: {abs_audio_file}")

        logger.debug(f"Transcribing audio file: {abs_audio_file} using whisper-cli")

        expected_json_output_file = abs_audio_file + ".json"

        cmd = [
            self.whisper_cli_path,
            "-m", self.model_path,
            "-f", abs_audio_file,
            "-oj", # Output JSON to <audio_file>.json
            "-l", "auto", # Auto-detect language
            "-nt", # No timestamps in output text (can make JSON smaller if not needed)
            "-np"  # No progress output to stdout/stderr from whisper-cli
        ]
        # Add other whisper.cpp flags as needed, e.g., -t <threads>, -p <processors>

        logger.debug(f"Executing whisper-cli with command: {' '.join(cmd)}")
        logger.debug(f"Expecting JSON output file at: {expected_json_output_file}")

        if os.path.exists(expected_json_output_file):
            try:
                os.remove(expected_json_output_file)
                logger.debug(f"Removed pre-existing JSON output file: {expected_json_output_file}")
            except OSError as e:
                logger.warning(f"Could not remove pre-existing JSON file {expected_json_output_file}: {e}")

        try:
            # Using DEVNULL for stdout/stderr as whisper-cli with -oj writes to file, and -np silences progress.
            # We only care about the return code here.
            process = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, # Capture stderr in case of errors not related to file output
                check=True, # Will raise CalledProcessError for non-zero exit codes
                encoding='utf-8'
            )
            # Log stderr if it contains anything, even on success (might have warnings)
            if process.stderr and process.stderr.strip():
                 logger.debug(f"--- whisper-cli STDERR (even on success) START ---\n{process.stderr.strip()}\n--- whisper-cli STDERR END ---")


            if not os.path.exists(expected_json_output_file):
                error_message = (
                    f"whisper-cli ran successfully (exit code 0), but the expected JSON output file was not found: {expected_json_output_file}\n"
                    f"Stderr from CLI: {process.stderr.strip() if process.stderr else 'N/A'}"
                )
                logger.error(error_message)
                raise FileNotFoundError(f"Expected JSON output file not created by whisper-cli: {expected_json_output_file}")

            logger.debug(f"Reading JSON output from: {expected_json_output_file}")
            with open(expected_json_output_file, 'r', encoding='utf-8') as f:
                result_json = json.load(f)

            full_text = ""
            # whisper.cpp JSON with -oj typically has a "transcription" list of segments
            if "transcription" in result_json and isinstance(result_json["transcription"], list):
                # Concatenate text from all segments
                segments_texts = [segment.get("text", "") for segment in result_json["transcription"]]
                full_text = "".join(segments_texts).strip() # Join segments, then strip whitespace
            # Fallback for other possible simple JSON structures (less likely with -oj)
            elif "text" in result_json:
                 full_text = str(result_json["text"]).strip()

            # If -nt is used, "text" inside segments should be clean.
            # If timestamps were present, segment["text"] might include them.
            # The main "text" field in the overall JSON might also exist with full transcription.

            final_result = {
                "text": full_text,
                "raw_whisper_cpp_output": result_json
            }
            logger.debug(f"Transcription completed successfully. Recognized text: \"{final_result['text']}\"")
            return final_result

        except subprocess.CalledProcessError as e:
            stderr_err = e.stderr.strip() if e.stderr else "N/A"
            error_message = (
                f"whisper-cli failed with exit code {e.returncode}.\n"
                f"Stderr: {stderr_err}"
            )
            logger.error(error_message)
            raise RuntimeError(f"Transcription failed. whisper-cli error. Check logs for details.") from e
        except json.JSONDecodeError as e_json:
            error_message = f"Failed to parse JSON from file {expected_json_output_file}.\nError: {e_json}"
            logger.error(error_message)
            try:
                with open(expected_json_output_file, 'r', encoding='utf-8') as f_err:
                    file_content_preview = f_err.read(1000)
                    logger.error(f"Content of problematic JSON file ({expected_json_output_file}) (first 1000 chars):\n{file_content_preview}...")
            except Exception as read_err:
                logger.error(f"Could not read problematic JSON file {expected_json_output_file} for debugging: {read_err}")
            raise RuntimeError(f"Failed to parse transcription output from file: {e_json}") from e_json
        except Exception as e_gen:
            logger.error(f"An unexpected error occurred during transcription: {e_gen}", exc_info=True)
            raise RuntimeError(f"An unexpected error occurred during transcription: {e_gen}") from e_gen
        finally:
            if os.path.exists(expected_json_output_file):
                try:
                    os.remove(expected_json_output_file)
                    logger.debug(f"Cleaned up JSON output file: {expected_json_output_file}")
                except OSError as e_os:
                    logger.warning(f"Could not remove JSON output file {expected_json_output_file}: {e_os}")

    def get_transcription_text(self, audio_file: str) -> str:
        """
        Get only the transcribed text from an audio file.

        Args:
            audio_file: Path to the audio file to transcribe

        Returns:
            Transcribed text as a string
        """
        result = self.transcribe(audio_file)
        return result["text"]