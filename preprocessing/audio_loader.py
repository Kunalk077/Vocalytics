"""
audio_loader.py

This module is responsible for loading audio files
using the Librosa library.
"""

import librosa
import os

from utils.constants import (
    SAMPLE_RATE,
    DURATION,
    OFFSET,
    SUPPORTED_AUDIO_FORMATS
)


def is_valid_audio_file(file_path):
    """
    Check if the uploaded file has a supported extension.

    Args:
        file_path (str): Path of the audio file.

    Returns:
        bool: True if supported, otherwise False.
    """
    _, extension = os.path.splitext(file_path)
    return extension.lower() in SUPPORTED_AUDIO_FORMATS


def load_audio(file_path):
    """
    Load an audio file using Librosa.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        tuple:
            audio (numpy.ndarray): Audio signal
            sample_rate (int): Sampling rate
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not is_valid_audio_file(file_path):
        raise ValueError("Unsupported audio format. Please upload a WAV file.")

    try:
        audio, sample_rate = librosa.load(
            file_path,
            sr=SAMPLE_RATE,
            duration=DURATION,
            offset=OFFSET
        )

        return audio, sample_rate

    except Exception as error:
        raise RuntimeError(f"Error loading audio: {error}")


if __name__ == "__main__":

    sample_file = "uploads/sample.wav"

    try:
        audio, sample_rate = load_audio(sample_file)

        print("Audio loaded successfully!")
        print(f"Sample Rate : {sample_rate}")
        print(f"Number of Samples : {len(audio)}")

    except Exception as error:
        print(error)