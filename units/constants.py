"""
constants.py

This file contains all the constant values used throughout the Vocalytics project.
Keeping constants in one place makes the project easier to maintain and update.
"""

# ==========================================
# Audio Settings
# ==========================================

# Sampling rate used while loading audio
SAMPLE_RATE = 22050

# Number of MFCC features to extract
N_MFCC = 40

# Duration (in seconds) to consider for each audio sample
DURATION = 3

# Offset (in seconds) from the beginning of the audio
OFFSET = 0.5


# ==========================================
# Model Paths
# ==========================================

MODEL_PATH = "models/emotion_model.pkl"
SCALER_PATH = "models/scaler.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"


# ==========================================
# Upload Folder
# ==========================================

UPLOAD_FOLDER = "uploads/"


# ==========================================
# Supported Audio Formats
# ==========================================

SUPPORTED_AUDIO_FORMATS = [".wav"]


# ==========================================
# Emotion Labels
# ==========================================

EMOTIONS = [
    "Angry",
    "Calm",
    "Disgust",
    "Fearful",
    "Happy",
    "Neutral",
    "Sad",
    "Surprised"
]


# ==========================================
# Random State
# ==========================================

# Used for reproducible model training
RANDOM_STATE = 42