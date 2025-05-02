import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import pyaudio
import wave
import numpy as np
import os
from datetime import datetime
import subprocess
import sys
import warnings
import requests
import json
import soundfile as sf
from scipy.io.wavfile import write as write_wav

# Filter out the specific FutureWarning from Whisper
warnings.filterwarnings("ignore", message="The input name `inputs` is deprecated.*")

# LMStudio Configuration
LMSTUDIO_API_URL = "http://localhost:1234/v1/chat/completions"  # Default LMStudio API endpoint
SYSTEM_PROMPT = "Ты виртуальный помощник-ассистент, будь дружелюбной и открытой к пользователю. Отвечай только текстом без смайликов"

# Check if ffmpeg is installed
try:
    subprocess.check_output(['ffmpeg', '-version'])
    print("ffmpeg is installed and working.")
except (subprocess.SubprocessError, FileNotFoundError):
    print("Error: ffmpeg is not installed or not in PATH.")
    print("Please install ffmpeg:")
    print("- On macOS: brew install ffmpeg")
    print("- On Ubuntu/Debian: sudo apt install ffmpeg")
    print("- On Windows: Download from https://www.ffmpeg.org/download.html and add to PATH")
    sys.exit(1)

# Parameters for recording
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper expects 16kHz audio
CHUNK = 1024
RECORD_SECONDS = 5  # Record for 5 seconds, adjust as needed

# Set device for models
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to use {device}")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load Whisper model and processor
print("Loading Whisper model...")
model_id = "openai/whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

# Create speech recognition pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    generate_kwargs={"max_new_tokens": 128},
    device=device
)

# Load Bark text-to-speech model
print("Loading Bark model...")
from transformers import BarkModel, BarkProcessor

bark_model = BarkModel.from_pretrained("suno/bark", torch_dtype=torch_dtype)
bark_processor = BarkProcessor.from_pretrained("suno/bark")
bark_model.to(device)

def record_audio():
    """Record audio from microphone and save to a temporary file"""
    print("Recording will start in 2 seconds...")
    import time
    time.sleep(2)

    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Start recording
    print("Recording... Speak now!")
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished!")

    # Stop recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio to a temporary file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_filename = f"temp_recording_{timestamp}.wav"

    wf = wave.open(temp_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    return temp_filename

def analyze_with_gemma(transcription):
    """Send the transcription to LMStudio's Gemma model for analysis"""
    try:
        payload = {
            "model": "gemma",  # This value might need to be adjusted based on LMStudio's configuration
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this transcribed speech: \"{transcription}\""}
            ],
            "temperature": 0.7,
            "max_tokens": 300
        }

        headers = {
            "Content-Type": "application/json"
        }

        response = requests.post(LMSTUDIO_API_URL, json=payload, headers=headers)

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"Error connecting to LMStudio: {response.status_code} - {response.text}"

    except Exception as e:
        return f"Error analyzing with Gemma: {str(e)}"

def text_to_speech_with_bark(text):
    """Convert text to speech using Bark"""
    try:
        # Process text
        inputs = bark_processor(
            text=text,
            voice_preset="v2/ru_speaker_0",  # You can change this to other voices
            return_tensors="pt",
        ).to(device)

        # Generate speech
        speech_output = bark_model.generate(**inputs, do_sample=True)
        audio_array = speech_output.cpu().numpy().squeeze()

        # Save and return the audio file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"response_{timestamp}.wav"

        # Bark sample rate is 24000 Hz
        sample_rate = bark_model.generation_config.sample_rate
        write_wav(output_filename, sample_rate, audio_array)

        return output_filename

    except Exception as e:
        print(f"Error generating speech: {str(e)}")
        return None

def play_audio(file_path):
    """Play audio file using system's default audio player"""
    try:
        if sys.platform == 'darwin':  # macOS
            subprocess.call(['afplay', file_path])
        elif sys.platform == 'linux':
            subprocess.call(['aplay', file_path])
        elif sys.platform == 'win32':
            os.system(f'start {file_path}')
        else:
            print("Unsupported OS for audio playback")
    except Exception as e:
        print(f"Error playing audio: {str(e)}")

def check_lmstudio_connection():
    """Check if LMStudio is running and accessible"""
    try:
        # Simple request to check if the server is running
        response = requests.get(LMSTUDIO_API_URL.rsplit('/', 1)[0])
        if response.status_code < 500:  # Any response that's not a server error
            return True
        return False
    except requests.exceptions.RequestException:
        return False

def main():
    print("\nVoice Assistant with Whisper, Gemma and Bark")
    print("-------------------------------------------")
    print("This program will:")
    print("1. Record your voice and transcribe it with Whisper")
    print("2. Analyze the transcription with Gemma")
    print("3. Read the analysis back to you with Bark")
    print(f"Recording duration: {RECORD_SECONDS} seconds")

    # Check LMStudio connection
    lmstudio_available = check_lmstudio_connection()
    if not lmstudio_available:
        print("\nWARNING: Could not connect to LMStudio API.")
        print("Make sure LMStudio is running and the Gemma model is loaded.")
        print("You should:")
        print("1. Open LMStudio")
        print("2. Load the Gemma model")
        print("3. Click on 'Local Server' and start the server")
        user_input = input("Continue without Gemma analysis? (y/n): ")
        if user_input.lower() != 'y':
            sys.exit(1)

    try:
        response_files = []  # Keep track of response files for cleanup

        while True:
            user_input = input("\nPress Enter to start recording (or type 'exit' to quit): ")
            if user_input.lower() == 'exit':
                break

            # Record audio from microphone
            audio_file = record_audio()

            print("Transcribing audio...")
            # Transcribe the recorded audio
            result = pipe(audio_file)
            transcription = result["text"]

            print("\nTranscription:")
            print(transcription)

            # Analyze with Gemma if LMStudio is available
            analysis = None
            if lmstudio_available:
                try:
                    print("\nAnalyzing with Gemma...")
                    analysis = analyze_with_gemma(transcription)
                    print("\nGemma Analysis:")
                    print(analysis)
                except Exception as e:
                    print(f"\nFailed to analyze with Gemma: {str(e)}")
                    analysis = f"I couldn't analyze that with Gemma. I heard you say: {transcription}"
            else:
                analysis = f"Gemma analysis is not available. I heard you say: {transcription}"

            # Generate speech response with Bark
            if analysis:
                print("\nGenerating speech with Bark...")
                response_file = text_to_speech_with_bark(analysis)
                if response_file:
                    response_files.append(response_file)
                    print("Playing response...")
                    play_audio(response_file)

            # Clean up recording file
            os.remove(audio_file)

    except KeyboardInterrupt:
        print("\nExiting program...")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up response files
        print("Cleaning up temporary files...")
        for file in response_files:
            try:
                if os.path.exists(file):
                    os.remove(file)
            except:
                pass

if __name__ == "__main__":
    main()