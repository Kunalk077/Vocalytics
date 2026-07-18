"""
feature_extractor.py

This module extracts MFCC features from an audio signal.
These features are used for training and prediction.
"""

import numpy as np
import librosa

from utils.constants import N_MFCC


def extract_mfcc(audio, sample_rate):
    """
    Extract MFCC features from an audio signal.

    Args:
        audio (numpy.ndarray): Audio signal.
        sample_rate (int): Sampling rate.

    Returns:
        numpy.ndarray:
            Mean MFCC feature vector.
    """

    try:
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=N_MFCC
        )

        mfcc = np.mean(mfcc.T, axis=0)

        return mfcc

    except Exception as error:
        raise RuntimeError(f"Error extracting MFCC features: {error}")


def extract_features(audio, sample_rate):
    """
    Extract all features required for prediction.

    Currently only MFCC features are used.
    This function is designed so additional features
    can be added later without changing other files.

    Args:
        audio (numpy.ndarray)
        sample_rate (int)

    Returns:
        numpy.ndarray
    """

    features = extract_mfcc(audio, sample_rate)

    return features


if __name__ == "__main__":

    from preprocessing.audio_loader import load_audio

    sample_file = "uploads/sample.wav"

    try:
        audio, sample_rate = load_audio(sample_file)

        features = extract_features(audio, sample_rate)

        print("Feature Extraction Successful")
        print(f"Feature Shape : {features.shape}")
        print(features)

    except Exception as error:
        print(error)