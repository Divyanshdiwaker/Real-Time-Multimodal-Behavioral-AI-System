# 🧠 Real-Time Multimodal Behavioral AI System

> An AI-powered behavioral analysis platform that combines facial expressions, vocal stress patterns, and speech content to generate real-time behavioral insights through multimodal deep learning.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📖 Overview

The **Real-Time Multimodal Behavioral AI System** analyzes human behavior using three independent modalities:

- 🎭 Facial Emotion Recognition
- 🎤 Voice Stress Analysis
- 💬 Speech Content Understanding

The extracted features are fused using a multimodal neural network to estimate:

- Stress Level
- Confidence Score
- Behavioral Incongruency

Unlike traditional deception systems, this project **does not detect lies**. Instead, it measures inconsistencies between visual, vocal, and linguistic signals.

---

## ✨ Key Features

### 🎭 Emotion Recognition

- Real-time webcam analysis
- CNN-based emotion classification
- 7 emotion categories
- 128-dimensional emotion embeddings

### 🎤 Voice Stress Detection

- Audio feature extraction using MFCCs
- LSTM-based stress prediction
- Real-time microphone analysis
- 64-dimensional stress embeddings

### 💬 Speech Understanding

- Speech-to-text transcription using Whisper
- Semantic sentence embeddings using Sentence Transformers
- Context-aware text representation

### 🔀 Multimodal Fusion

- Gated Fusion Network
- Cross-modal behavioral analysis
- Stress estimation
- Behavioral incongruency detection

### 📊 Session Analytics

- Session logging
- Performance tracking
- Automatic PDF report generation
- Visual analytics dashboard

---

# 🏗️ System Architecture

```text
 Webcam Video
       │
       ▼
 Emotion CNN
 (138 Features)
       │
       │
       ▼

 Microphone Audio
       │
       ▼
 Stress LSTM
 (64 Features)

       │
       ▼

 Speech Audio
       │
       ▼
 Whisper ASR
       │
       ▼
 Sentence Transformer
 (128 Features)

       │
       ▼

 ┌───────────────────┐
 │  Fusion Network   │
 └───────────────────┘

       │
       ▼

 ┌───────────────────┐
 │ Stress Score      │
 │ Confidence Score  │
 │ Incongruency      │
 └───────────────────┘
```

---

## 🧠 Model Architecture

### Emotion Model

| Component      | Details                                             |
| -------------- | --------------------------------------------------- |
| Architecture   | CNN                                                 |
| Input          | 48×48 Grayscale Face                                |
| Classes        | Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral |
| Embedding Size | 138                                                 |
*Added 10 embeddings for facial landmarks module, gives values for Gaze eye, Head Tilt Angle, Mouth Opening, Brow Raise

---

### Stress Model

| Component      | Details |
| -------------- | ------- |
| Architecture   | LSTM    |
| Features       | MFCC    |
| Hidden Size    | 64      |
| Embedding Size | 64      |

---

### Text Model

| Component      | Details                    |
| -------------- | -------------------------- |
| Speech-to-Text | Whisper                    |
| Embeddings     | SentenceTransformer MiniLM |
| Embedding Size | 128                        |

---

### Fusion Model

| Component      | Details                          |
| -------------- | -------------------------------- |
| Input Features | 330                              |
| Fusion Method  | Gated Fusion                     |
| Outputs        | Stress + Behavioral Incongruency |

---

## 📂 Project Structure

```text
Real-Time-Multimodal-Behavioral-AI-System
│
├── backend
│   ├── models
│   ├── routes
│   ├── services
│   ├── session
│   └── utils
│
├── frontend
│   └── app.py
│
├── Training
│   ├── train_emotion.py
│   ├── train_stress.py
│   ├── train_fusion.py
│   ├── generate_embeddings.py
│   └── generate_fusion_dataset.py
│
├── Data
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🛠️ Tech Stack

### Backend

- FastAPI
- PyTorch
- OpenCV
- Librosa
- NumPy
- Pandas

### Frontend

- Streamlit

### NLP

- Whisper
- Sentence Transformers

### Visualization & Reporting

- Matplotlib
- ReportLab

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/Divyanshdiwaker/Real-Time-Multimodal-Behavioral-AI-System.git
cd Real-Time-Multimodal-Behavioral-AI-System
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate:

Windows

```bash
venv\Scripts\activate
```

Linux/Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Start 

```bash
python -m streamlit run frontend/app.py
```

---

## 📊 Output Metrics

The system generates:

| Metric                  | Description                       |
| ----------------------- | --------------------------------- |
| Stress Level            | Estimated stress probability      |
| Confidence Score        | Inverse stress indicator          |
| Behavioral Incongruency | Cross-modal mismatch score        |
| Session Report          | PDF summary with visual analytics |

---



## 🎯 Applications

- Human Behavior Analysis
- Interview Assessment Research
- Educational AI Systems
- Human-Computer Interaction
- Behavioral Analytics Research
- Multimodal AI Research

---

## 🔮 Future Enhancements

- Transformer-Based Fusion
- ONNX Deployment
- Cloud API Deployment
- User Authentication
- Historical Trend Analysis
- Real-Time Alerts

---

## 📚 Datasets

### Emotion Recognition

- FER-2013 Dataset

### Voice Stress Detection

- RAVDESS Dataset

---

## 👨‍💻 Author

### Divyansh Diwaker

B.Tech (Artificial Intelligence & Machine Learning)

Guru Gobind Singh Indraprastha University (GGSIPU)

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
#
