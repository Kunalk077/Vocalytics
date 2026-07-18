"""
waveform.py

This module generates and displays
the waveform of an audio file.
"""

import matplotlib.pyplot as plt
import librosa
import librosa.display


def plot_waveform(audio, sample_rate):
    """
    Plot the waveform of an audio signal.

    Args:
        audio (numpy.ndarray): Audio signal.
        sample_rate (int): Sampling rate.

    Returns:
        matplotlib.figure.Figure
    """

    fig, ax = plt.subplots(figsize=(10, 4))

    librosa.display.waveshow(
        audio,
        sr=sample_rate,
        ax=ax
    )

    ax.set_title("Audio Waveform")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")

    plt.tight_layout()

    return fig


if __name__ == "__main__":

    from preprocessing.audio_loader import load_audio

    sample_file = "uploads/sample.wav"

    try:

        audio, sample_rate = load_audio(sample_file)

        figure = plot_waveform(audio, sample_rate)

        plt.show()

    except Exception as error:
        print(error)