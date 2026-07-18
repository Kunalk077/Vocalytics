"""
save_model.py

This module provides utility functions to save
and load machine learning models.
"""

import joblib


def save_model(model, scaler, label_encoder,
               model_path,
               scaler_path,
               label_encoder_path):
    """
    Save trained model, scaler and label encoder.
    """

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, label_encoder_path)

    print("Model saved successfully.")


def load_model(model_path,
               scaler_path,
               label_encoder_path):
    """
    Load trained model, scaler and label encoder.
    """

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)

    return model, scaler, label_encoder