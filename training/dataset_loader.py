"""
dataset_loader.py

This module loads the RAVDESS dataset, extracts MFCC features,
and prepares the feature matrix (X) and labels (y) for training.
"""

import os
import numpy as np

from preprocessing.audio_loader import load_audio
from preprocessing.feature_extractor import extract_features


# Mapping of RAVDESS emotion codes to emotion names
EMOTION_MAP = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised"
}


def get_emotion_label(filename):
    """
    Extract emotion label from RAVDESS filename.

    Example:
    03-01-05-01-01-01-01.wav

    Emotion code = 05
    Emotion = Angry
    """

    parts = filename.split("-")

    if len(parts) < 3:
        return None

    emotion_code = parts[2]

    return EMOTION_MAP.get(emotion_code)


def load_dataset(dataset_path):
    """
    Load the complete dataset.

    Args:
        dataset_path (str)

    Returns:
        X : numpy.ndarray
        y : numpy.ndarray
    """

    X = []
    y = []

    print("Loading dataset...")

    for root, _, files in os.walk(dataset_path):

        for file in files:

            if not file.endswith(".wav"):
                continue

            file_path = os.path.join(root, file)

            emotion = get_emotion_label(file)

            if emotion is None:
                continue

            try:
                audio, sample_rate = load_audio(file_path)

                features = extract_features(audio, sample_rate)

                X.append(features)
                y.append(emotion)

            except Exception as error:
                print(f"Skipping {file}: {error}")

    print(f"Loaded {len(X)} audio files.")

    return np.array(X), np.array(y)


if __name__ == "__main__":

    DATASET_PATH = "dataset/ravdess"

    X, y = load_dataset(DATASET_PATH)

    print("\nDataset Loaded Successfully")
    print(f"Feature Shape : {X.shape}")
    print(f"Labels Shape  : {y.shape}")

    print("\nSample Labels:")
    print(y[:10])