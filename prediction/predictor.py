"""
predictor.py

This module loads the trained model and predicts
the emotion from an input audio file.
"""
import os

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found. Please train the model first."
    )
import joblib
import numpy as np

from preprocessing.audio_loader import load_audio
from preprocessing.feature_extractor import extract_features
from utils.constants import (
    MODEL_PATH,
    SCALER_PATH,
    LABEL_ENCODER_PATH
)


class EmotionPredictor:
    """
    Emotion Predictor class for loading the trained model
    and making predictions.
    """

    def __init__(self):
        """Load trained model, scaler and label encoder."""

        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.label_encoder = joblib.load(LABEL_ENCODER_PATH)

    def predict(self, audio_path):
        """
        Predict emotion from an audio file.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            tuple:
                emotion (str)
                confidence (float)
        """

        # Load audio
        audio, sample_rate = load_audio(audio_path)

        # Extract features
        features = extract_features(audio, sample_rate)

        # Reshape for prediction
        features = features.reshape(1, -1)

        # Scale features
        features = self.scaler.transform(features)

        # Predict emotion
        prediction = self.model.predict(features)

        # Predict probability
        probabilities = self.model.predict_proba(features)

        confidence = np.max(probabilities) * 100

        emotion = self.label_encoder.inverse_transform(prediction)[0]

        return emotion, confidence


if __name__ == "__main__":

    predictor = EmotionPredictor()

    sample_audio = "uploads/sample.wav"

    try:
        emotion, confidence = predictor.predict(sample_audio)

        print("=" * 50)
        print("Prediction Successful")
        print("=" * 50)
        print(f"Predicted Emotion : {emotion}")
        print(f"Confidence         : {confidence:.2f}%")

    except Exception as error:
        print(error)

