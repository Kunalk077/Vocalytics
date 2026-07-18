"""
app.py

Main Streamlit application for Vocalytics.
"""

import os
import streamlit as st

from preprocessing.audio_loader import load_audio
from prediction.predictor import EmotionPredictor
from visualization.waveform import plot_waveform
from visualization.spectrogram import plot_spectrogram


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Vocalytics",
    page_icon="🎙️",
    layout="wide"
)

# ==========================================
# Title
# ==========================================

st.title("🎙️ Vocalytics")
st.subheader("Speech Emotion Detection System")

st.markdown("---")

# ==========================================
# Sidebar
# ==========================================

st.sidebar.title("About")

st.sidebar.info(
    """
    Vocalytics is a Speech Emotion Detection System.

    Workflow:

    Upload Audio
        ↓
    Feature Extraction
        ↓
    Random Forest Model
        ↓
    Emotion Prediction
    """
)

st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "Upload WAV File",
    type=["wav"]
)

# ==========================================
# Load Predictor
# ==========================================

predictor = EmotionPredictor()

# ==========================================
# Process Audio
# ==========================================

if uploaded_file is not None:

    upload_folder = "uploads"

    os.makedirs(upload_folder, exist_ok=True)

    file_path = os.path.join(upload_folder, uploaded_file.name)

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    st.success("Audio uploaded successfully.")

    st.audio(file_path)

    # Load Audio
    audio, sample_rate = load_audio(file_path)

    # --------------------------------------
    # Waveform
    # --------------------------------------

    st.subheader("Waveform")

    waveform = plot_waveform(audio, sample_rate)

    st.pyplot(waveform)

    # --------------------------------------
    # Spectrogram
    # --------------------------------------

    st.subheader("Spectrogram")

    spectrogram = plot_spectrogram(audio, sample_rate)

    st.pyplot(spectrogram)

    # --------------------------------------
    # Prediction
    # --------------------------------------

    if st.button("Predict Emotion"):

        with st.spinner("Predicting emotion..."):

            emotion, confidence = predictor.predict(file_path)

        st.markdown("---")

        st.success("Prediction Completed")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Predicted Emotion",
                value=emotion
            )

        with col2:
            st.metric(
                label="Confidence",
                value=f"{confidence:.2f}%"
            )

        st.balloons()

else:

    st.info("Please upload a WAV audio file to begin.")