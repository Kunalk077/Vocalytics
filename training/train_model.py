"""
train_model.py

This module trains a Random Forest classifier for
speech emotion detection using extracted MFCC features.
"""

import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from training.dataset_loader import load_dataset
from utils.constants import (
    RANDOM_STATE,
    MODEL_PATH,
    SCALER_PATH,
    LABEL_ENCODER_PATH
)


def train_model(dataset_path):
    """
    Train the emotion detection model.

    Args:
        dataset_path (str): Path to RAVDESS dataset.
    """

    print("=" * 60)
    print("Loading Dataset...")
    print("=" * 60)

    X, y = load_dataset(dataset_path)

    print(f"\nTotal Samples : {len(X)}")
    print(f"Feature Shape : {X.shape}")

    # ---------------------------------------
    # Encode Labels
    # ---------------------------------------

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # ---------------------------------------
    # Scale Features
    # ---------------------------------------

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---------------------------------------
    # Train-Test Split
    # ---------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_encoded,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )

    print(f"\nTraining Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")

    # ---------------------------------------
    # Train Model
    # ---------------------------------------

    model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=2,
    random_state=RANDOM_STATE
)

    print("\nTraining Model...\n")

    model.fit(X_train, y_train)

    # ---------------------------------------
    # Predictions
    # ---------------------------------------

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(f"\nAccuracy : {accuracy * 100:.2f}%")

    print("\nClassification Report\n")
    print(classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_
    ))

    print("\nConfusion Matrix\n")
    print(confusion_matrix(y_test, predictions))

    # ---------------------------------------
    # Save Model
    # ---------------------------------------

    from training.save_model import save_model

save_model(
    model,
    scaler,
    label_encoder,
    MODEL_PATH,
    SCALER_PATH,
    LABEL_ENCODER_PATH
)

    print("\nModel saved successfully!")
    print(f"Model Path : {MODEL_PATH}")


if __name__ == "__main__":

    DATASET_PATH = "dataset/ravdess"

    train_model(DATASET_PATH)