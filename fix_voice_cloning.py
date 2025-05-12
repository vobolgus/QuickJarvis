#!/usr/bin/env python
"""
This script fixes voice cloning by downgrading PyTorch to a version
compatible with the TTS library.
"""
import subprocess
import sys
import os
import pkg_resources


def get_installed_version(package_name):
    """Get the installed version of a package"""
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return None


def main():
    print("\n===== Voice Cloning Compatibility Fix =====\n")

    # Check current versions
    torch_version = get_installed_version("torch")
    tts_version = get_installed_version("TTS")

    print(f"Current versions:\n- PyTorch: {torch_version}\n- TTS: {tts_version}\n")
    print("The error you're experiencing is due to incompatibility between PyTorch and TTS.")
    print("We need to downgrade to compatible versions.\n")

    # Ask for confirmation
    choice = input("Do you want to proceed with fixing this issue? (y/n): ").strip().lower()
    if choice != 'y':
        print("Operation cancelled.")
        return

    print("\nStep 1: Uninstalling current packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "TTS"])
        print("✅ Uninstallation successful")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Error during uninstallation: {e}")
        print("Continuing anyway...")

    print("\nStep 2: Installing compatible versions...")
    try:
        # Install PyTorch 2.0.1 - this is known to work with TTS
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "torch==2.0.1", "torchvision==0.15.2", "torchaudio==2.0.2"])
        print("✅ PyTorch 2.0.1 installed successfully")

        # Install TTS 0.21.0
        subprocess.check_call([sys.executable, "-m", "pip", "install", "TTS==0.21.0"])
        print("✅ TTS 0.21.0 installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        print("Please try manually running:")
        print("pip uninstall -y torch torchvision torchaudio TTS")
        print("pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2")
        print("pip install TTS==0.21.0")
        return

    # Verify installation
    new_torch_version = get_installed_version("torch")
    new_tts_version = get_installed_version("TTS")

    print(f"\nNew versions:\n- PyTorch: {new_torch_version}\n- TTS: {new_tts_version}\n")

    if new_torch_version == "2.0.1" and new_tts_version == "0.21.0":
        print("✅ Success! You now have compatible versions installed.")
        print("\nNow you can try voice cloning again. Run your voice assistant with:")
        print("python main.py")
    else:
        print("⚠️ Warning: The installed versions are not exactly as expected.")
        print("Voice cloning might still work, but if you encounter issues, try manual installation.")

    print("\nIf you experience any issues with other functionality after this change,")
    print("you can run 'pip install -r requirements.txt' to reinstall other dependencies.")


if __name__ == "__main__":
    main()