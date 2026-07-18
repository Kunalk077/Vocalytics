"""
spectrogram.py

This module generates and displays
the spectrogram of an audio signal.
"""

import matplotlib.pyplot as plt
import librosa
import librosa.display


def plot_spectrogram(audio, sample_rate):
    """
    Plot the spectrogram of an audio signal.

    Args:
        audio (numpy.ndarray): Audio signal.
        sample_rate (int): Sampling rate.

    Returns:
        matplotlib.figure.Figure
    """

    # Compute Short-Time Fourier Transform (STFT)
    stft = librosa.stft(audio)

    # Convert amplitude to decibels
import numpy as np
spectrogram = librosa.amplitude_to_db(
    np.abs(stft),
    ref=np.max
)

    # Create plot
fig, ax = plt.subplots(figsize=(10, 4))

librosa.display.specshow(
        spectrogram,
        sr=sample_rate,
        x_axis="time",
        y_axis="hz",
        cmap="magma",
        ax=ax
    )

fig.colorbar(
        ax.images[0],
        ax=ax,
        format="%+2.0f dB"
    )

ax.set_title("Audio Spectrogram")

plt.tight_layout()
return fig

if __name__ == "__main__":

    from preprocessing.audio_loader import load_audio

    sample_file = "uploads/sample.wav"

    try:

        audio, sample_rate = load_audio(sample_file)

        figure = plot_spectrogram(audio, sample_rate)

        plt.show()

    except Exception as error:
        print(error)