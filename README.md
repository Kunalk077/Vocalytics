<div align="center">

<img src="assets/banner.png" alt="Vocalytics Banner" width="100%">

# 🎙️ Vocalytics

### *Speech Emotion Detection System using Machine Learning*

<p align="center">
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/Librosa-AI%20Audio-blueviolet?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
<img src="https://img.shields.io/badge/Random%20Forest-228B22?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Status-Active-00FF99?style=for-the-badge"/>
</p>

*"Transforming Human Voice into Emotional Intelligence."*

</div>

---

# 🚀 Overview

**Vocalytics** is a Speech Emotion Detection System that predicts the emotion expressed in human speech using Machine Learning.

The application allows users to upload a WAV audio file, extracts MFCC features using **Librosa**, and predicts emotions using a trained **Random Forest Classifier**. Results are presented through a clean and interactive **Streamlit** dashboard.

---

# ⚡ Features

✨ Upload WAV audio files

🎵 Automatic audio preprocessing

📊 MFCC Feature Extraction

🧠 Random Forest based emotion prediction

📈 Confidence score display

🌊 Audio waveform visualization

🎼 Spectrogram visualization

🖥️ Interactive Streamlit interface

📦 Modular project architecture

---

# 🧠 Workflow

```text
              +---------------------+
              | Upload Audio (.wav) |
              +----------+----------+
                         |
                         ▼
              +---------------------+
              |   Load Audio File   |
              +----------+----------+
                         |
                         ▼
              +---------------------+
              | Extract MFCC Features|
              +----------+----------+
                         |
                         ▼
              +---------------------+
              | Random Forest Model |
              +----------+----------+
                         |
                         ▼
              +---------------------+
              | Predict Emotion     |
              +----------+----------+
                         |
                         ▼
              +---------------------+
              | Display Result      |
              +---------------------+
```

---

# 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Audio Processing | Librosa |
| Web Framework | Streamlit |
| Visualization | Matplotlib |
| Data Handling | NumPy, Pandas |
| Model Storage | Joblib |

---

# 📂 Project Structure

```text
Vocalytics
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── assets/
├── dataset/
├── models/
├── prediction/
├── preprocessing/
├── training/
├── uploads/
├── utils/
├── visualization/
└── notebooks/
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/your-username/Vocalytics.git
```

Go to project folder

```bash
cd Vocalytics
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 📊 Model Pipeline

- Load audio using Librosa
- Extract MFCC features
- Normalize features
- Predict emotion using Random Forest
- Display prediction with confidence
- Visualize waveform and spectrogram

---

# 🎯 Supported Emotions

- 😀 Happy
- 😢 Sad
- 😠 Angry
- 😨 Fear
- 😐 Neutral
- 😌 Calm
- 🤢 Disgust
- 😲 Surprise

---

# 📸 Screenshots

> *(Screenshots will be added after project completion.)*

```
assets/screenshots/
```

---

# 🔮 Future Improvements

- Deep Learning (CNN/LSTM)
- Live microphone input
- Real-time emotion detection
- Additional datasets
- Cloud deployment
- User authentication
- Emotion analytics dashboard

---

# 🤝 Contributing

Contributions are welcome.

Feel free to fork this repository, create a new branch, and submit a Pull Request.

---

# 📜 License

This project is intended for educational and portfolio purposes.

---

<div align="center">

## 👨‍💻 Developer

### **Kunal Tyagi**

**Software Developer • Backend Developer • Machine Learning Enthusiast**

⭐ If you found this project useful, consider giving it a star.

*"Code. Learn. Build. Repeat."*

</div>