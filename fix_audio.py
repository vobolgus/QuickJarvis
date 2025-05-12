"""
This script diagnoses and fixes audio playback issues in the Voice Assistant.
It will check your audio configuration and test playback functionality.
"""
import os
import sys
import platform
import subprocess
import time
import tempfile
import wave
import array
import math


def create_test_audio():
    """Create a simple test beep audio file"""
    print("Creating test audio file...")

    # Create a simple 1-second 440Hz beep
    duration = 1  # seconds
    frequency = 440  # Hz (A4 note)
    sample_rate = 16000  # samples per second
    amplitude = 32000  # max amplitude

    # Generate sine wave
    num_samples = duration * sample_rate
    samples = array.array('h', [int(amplitude * math.sin(2 * math.pi * frequency * t / sample_rate))
                                for t in range(num_samples)])

    # Save as WAV file
    test_file = os.path.join(tempfile.gettempdir(), "test_beep.wav")
    with wave.open(test_file, 'w') as fp:
        fp.setnchannels(1)
        fp.setsampwidth(2)
        fp.setframerate(sample_rate)
        fp.writeframes(samples.tobytes())

    print(f"Test audio created at: {test_file}")
    return test_file


def check_audio_device():
    """Check if audio output devices are available and working"""
    print("\nChecking audio output devices...")
    system = platform.system().lower()

    if system == "linux":
        try:
            output = subprocess.check_output(["aplay", "-l"], universal_newlines=True)
            print("Found audio devices:")
            print(output)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            try:
                # Alternative check using pulseaudio
                output = subprocess.check_output(["pactl", "list", "sinks"], universal_newlines=True)
                print("Found PulseAudio sinks:")
                print(output)
                return True
            except (subprocess.SubprocessError, FileNotFoundError):
                print("⚠️ Could not detect audio devices with aplay or pactl")
                return False

    elif system == "darwin":  # macOS
        try:
            output = subprocess.check_output(["system_profiler", "SPAudioDataType"], universal_newlines=True)
            print("Audio information:")
            # Just show output devices
            for line in output.splitlines():
                if "Output" in line or "Speakers" in line or "Headphones" in line:
                    print(line.strip())
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            print("⚠️ Could not detect audio devices with system_profiler")
            return False

    elif system == "windows":
        # On Windows, it's harder to list audio devices from command line
        # PowerShell has ways but they're complex for a quick script
        print("On Windows, we'll test playback directly instead of listing devices")
        return True

    return False


def test_playback(test_file):
    """Test audio playback"""
    print("\nTesting audio playback...")
    system = platform.system().lower()

    try:
        if system == "darwin":  # macOS
            subprocess.run(["afplay", test_file], check=True)
        elif system == "linux":
            subprocess.run(["aplay", test_file], check=True)
        elif system == "windows":
            # On Windows, start command should launch default audio player
            os.system(f'start "" "{os.path.abspath(test_file)}"')
            # Give some time for the audio to play
            time.sleep(2)
        else:
            print(f"Unsupported OS: {system}")
            return False

        print("✅ Audio playback test completed")
        return True
    except Exception as e:
        print(f"❌ Audio playback test failed: {e}")
        return False


def check_playback_module():
    """Check the playback.py module for issues"""
    print("\nChecking playback.py module...")

    if not os.path.exists("playback.py"):
        print("❌ playback.py file not found!")
        return False

    with open("playback.py", "r") as f:
        content = f.read()

    # Check for common issues
    issues = []

    if "subprocess.DEVNULL" not in content and "subprocess.PIPE" not in content:
        issues.append("- May not be properly redirecting audio command output")

    if "os.path.abspath" not in content and platform.system().lower() == "windows":
        issues.append("- Windows playback might need absolute paths")

    # Check for platform-specific implementations
    platforms = []
    if "sys.platform == 'darwin'" in content:
        platforms.append("macOS")
    if "sys.platform == 'linux'" in content:
        platforms.append("Linux")
    if "sys.platform == 'win32'" in content:
        platforms.append("Windows")

    current_platform = platform.system()
    if (current_platform.lower() == "darwin" and "macOS" not in platforms) or \
            (current_platform.lower() == "linux" and "Linux" not in platforms) or \
            (current_platform.lower() == "windows" and "Windows" not in platforms):
        issues.append(f"- Missing implementation for your platform ({current_platform})")

    if issues:
        print("Found potential issues in playback.py:")
        for issue in issues:
            print(issue)
    else:
        print("✅ playback.py looks good")

    return len(issues) == 0


def fix_playback_module():
    """Create a fixed version of the playback module"""
    print("\nCreating improved playback.py...")

    improved_playback = """\"\"\"
Audio playback functionality for the Voice Assistant.
Improved for wider compatibility across platforms.
\"\"\"
import os
import subprocess
import sys
import time
from typing import Optional

from utils import logger


def play_audio(file_path: str) -> bool:
    \"\"\"
    Play an audio file using the system's default audio player.
    Enhanced with better error handling and platform detection.

    Args:
        file_path: Path to the audio file to play

    Returns:
        True if playback was successful, False otherwise
    \"\"\"
    if not os.path.exists(file_path):
        logger.error(f"Audio file not found: {file_path}")
        return False

    # Ensure we're using an absolute path to avoid platform-specific issues
    abs_file_path = os.path.abspath(file_path)

    logger.debug(f"Playing audio file: {abs_file_path}")

    # Flag to indicate playback attempt result
    playback_success = False

    try:
        # Use platform-specific commands for audio playback
        if sys.platform == 'darwin':  # macOS
            # Use subprocess.run for better timeout management
            process = subprocess.run(
                ['afplay', abs_file_path], 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.PIPE,
                check=True,
                timeout=30  # Set a reasonable timeout
            )
            playback_success = True
            logger.debug(f"macOS afplay completed with return code {process.returncode}")

        elif sys.platform == 'linux':  # Linux
            # Try multiple players in order of preference
            linux_players = [
                ['aplay', '-q', abs_file_path],
                ['paplay', abs_file_path],
                ['mplayer', '-really-quiet', abs_file_path],
                ['mpg123', '-q', abs_file_path]
            ]

            for player_cmd in linux_players:
                try:
                    logger.debug(f"Trying Linux player: {player_cmd[0]}")
                    process = subprocess.run(
                        player_cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        check=True,
                        timeout=30
                    )
                    playback_success = True
                    logger.debug(f"Linux player {player_cmd[0]} completed with return code {process.returncode}")
                    break  # Exit loop on successful player
                except (subprocess.SubprocessError, FileNotFoundError) as e:
                    logger.debug(f"Player {player_cmd[0]} failed: {e}")
                    continue  # Try next player

            if not playback_success:
                logger.warning("All Linux audio players failed, falling back to xdg-open")
                try:
                    # Last resort - use xdg-open to open with default application
                    subprocess.run(['xdg-open', abs_file_path], check=False)
                    playback_success = True
                except Exception as e:
                    logger.warning(f"xdg-open failed: {e}")

        elif sys.platform == 'win32':  # Windows
            try:
                # First try with winsound module for most reliable playback
                import winsound
                # NOTE: winsound.PlaySound expects a Windows-style path
                winsound.PlaySound(abs_file_path, winsound.SND_FILENAME)
                playback_success = True
                logger.debug("Windows winsound playback completed")
            except (ImportError, RuntimeError) as e:
                logger.debug(f"winsound failed: {e}, trying alternative methods")
                try:
                    # Fall back to media player if winsound fails
                    # rundll32 is more reliable than 'start' for audio files
                    os.system(f'rundll32 winmm.dll,PlaySound "{abs_file_path}" 0')
                    playback_success = True
                    logger.debug("Windows rundll32 playback initiated")
                except Exception as e2:
                    logger.debug(f"rundll32 failed: {e2}, trying start command")
                    try:
                        # Last resort - use start command
                        os.system(f'start /MIN "" "{abs_file_path}"')
                        playback_success = True
                        logger.debug("Windows start command initiated")
                    except Exception as e3:
                        logger.error(f"Windows playback failed with all methods: {e3}")

        else:
            logger.warning(f"Unsupported OS for audio playback: {sys.platform}")
            # Try a generic approach with Python's webbrowser module
            try:
                import webbrowser
                webbrowser.open(abs_file_path)
                playback_success = True
                logger.debug("Generic playback initiated via webbrowser module")
            except Exception as e:
                logger.error(f"Generic playback attempt failed: {e}")

        # If we've reached here without setting playback_success to True,
        # then all methods have failed
        if not playback_success:
            logger.error("All playback methods failed")
            return False

        return True

    except Exception as e:
        logger.error(f"Error playing audio: {str(e)}")
        return False
"""

    # Backup original if it exists
    if os.path.exists("playback.py"):
        backup_path = "playback.py.backup"
        i = 1
        while os.path.exists(backup_path):
            backup_path = f"playback.py.backup{i}"
            i += 1

        try:
            with open("playback.py", "r") as src, open(backup_path, "w") as dst:
                dst.write(src.read())
            print(f"Original playback.py backed up to {backup_path}")
        except Exception as e:
            print(f"Warning: Could not create backup: {e}")

    # Write improved version
    try:
        with open("playback.py", "w") as f:
            f.write(improved_playback)
        print("✅ Created improved playback.py")
        return True
    except Exception as e:
        print(f"❌ Failed to write improved playback.py: {e}")
        return False


def main():
    print("===== Voice Assistant Audio Playback Diagnostic =====")

    # Check audio devices
    devices_ok = check_audio_device()
    if not devices_ok:
        print("\n⚠️ Warning: Could not properly detect audio devices.")
        print("This might be normal on some systems, continuing with tests.")

    # Create test audio
    test_file = create_test_audio()

    # Check playback module
    module_ok = check_playback_module()

    # Test playback
    playback_ok = test_playback(test_file)

    print("\n===== Diagnostic Results =====")
    print(f"Audio devices check: {'✅ Passed' if devices_ok else '⚠️ Issues detected'}")
    print(f"Playback module check: {'✅ Looks good' if module_ok else '⚠️ Issues detected'}")
    print(f"Audio playback test: {'✅ Passed' if playback_ok else '❌ Failed'}")

    if not playback_ok or not module_ok:
        print("\nWould you like to install an improved version of the playback module? (y/n)")
        choice = input("> ").strip().lower()
        if choice == 'y':
            if fix_playback_module():
                print("\n✅ Playback module has been improved.")
                print("Please restart your voice assistant to apply the changes.")
            else:
                print("\n❌ Failed to install improved playback module.")
                print("You may need to fix it manually.")
    else:
        print("\n✅ Audio playback appears to be working correctly.")
        print("If you're still experiencing issues, here are some things to check:")
        print("1. Make sure your system volume is not muted")
        print("2. Verify your speakers or headphones are connected properly")
        print("3. Check if other applications can play sound")
        print("4. Ensure the voice assistant has permission to access audio devices")

    # Clean up
    try:
        os.remove(test_file)
        print(f"\nRemoved test audio file: {test_file}")
    except Exception:
        pass


if __name__ == "__main__":
    main()