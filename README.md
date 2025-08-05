# 🧠 Mental Fitness Tracker using SVM

<p align="center">
  <img src="https://img.icons8.com/external-flaticons-lineal-color-flat-icons/64/null/external-mental-health-healthcare-and-medical-flaticons-lineal-color-flat-icons.png" width="80"/>
</p>

A comprehensive, AI-powered web application designed to **monitor mental well-being** by analyzing both **text and speech inputs** using a robust **Support Vector Machine (SVM)** model.

---

## 📌 Table of Contents

* [🚀 Overview](#-overview)
* [✨ Features](#-features)
* [🧰 Tech Stack](#-tech-stack)
* [📂 Project Structure](#-project-structure)
* [🧠 Methodology](#-methodology)
* [🧪 Results](#-results)
* [📥 Installation](#-installation)
* [💡 Future Enhancements](#-future-enhancements)
* [🤝 Contributors](#-contributors)

---

## 🚀 Overview

The **Mental Fitness Tracker** utilizes **machine learning**, **natural language processing**, and **speech emotion recognition** to detect signs of stress, anxiety, or emotional fatigue. By accepting user input through text or speech, the model classifies the emotional state and provides **customized wellness recommendations**.

> "Mental health isn't just the absence of mental illness, but a proactive approach to well-being."

---

## ✨ Features

* 🔤 **Text-Based Emotion Analysis** using NLP & TF-IDF
* 🔊 **Speech Emotion Recognition** using MFCC audio features
* 🧪 Real-time **SVM Classification**
* 📊 Accuracy: **\~90%** | Precision: **88%** | Recall: **91%**
* 🧘 Personalized **Wellness Suggestions** (e.g., breathing exercises, music, motivational quotes)
* 🌐 Web Interface with Voice/Text Input Support

---

## 🧰 Tech Stack

### 💻 Frontend

* HTML5, CSS3, JavaScript
* Bootstrap 5 (Responsive UI)

### ⚙️ Backend

* Python 3.10+
* Flask (REST API)
* SQLite / MySQL

### 🤖 ML & NLP

* Scikit-learn (SVM, TF-IDF)
* Librosa (Audio feature extraction)
* NLTK / spaCy (Text preprocessing)

### 📦 Others

* Google Speech Recognition / Mozilla DeepSpeech
* Joblib (Model serialization)
* Git & GitHub (Version Control)
* Docker & Heroku (for deployment)

---

## 📂 Project Structure

```bash
MENTAL-FITNESS-TRACKER-USING-SVM/
├── static/                    # Images, audio files
├── templates/                # HTML templates (Flask)
├── app.py                    # Flask application entry point
├── model/
│   ├── svm_text_model.pkl
│   └── svm_audio_model.pkl
├── utils/
│   ├── audio_processor.py
│   └── text_processor.py
├── requirements.txt
└── README.md
```

---

## 🧠 Methodology

### 🎯 Input Modalities

* **Text Input**: Typed journal entries, messages
* **Speech Input**: Audio captured through microphone

### 🛠️ Preprocessing

* Text: Lowercasing, stopword removal, lemmatization
* Speech: MFCCs, pitch, tempo, spectral features

### 🧪 Feature Extraction

* TF-IDF for text
* Librosa for speech features (e.g., MFCC, chroma, spectral contrast)

### 🤖 Classification

* Binary classification ("Stressed" vs. "Not Stressed")
* SVM (RBF Kernel)

### 🎯 Output

* **Emotion Detected**
* **Confidence Score**
* **Action Recommendation** (meditation, quotes, music, etc.)

---

## 🧪 Results

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 90.3% |
| Precision | 88.1% |
| Recall    | 91.7% |
| F1-Score  | 89.9% |

📊 **Sample Input & Output**

| Input Type | Example                          | Detected Emotion | Suggested Action         |
| ---------- | -------------------------------- | ---------------- | ------------------------ |
| Text       | "I'm feeling low and exhausted." | Stressed         | Show relaxation video    |
| Voice      | Slow, trembling tone             | Stressed         | Suggest deep breathing   |
| Text       | "Everything's fine today!"       | Not Stressed     | Reinforce positive state |

---

## 📥 Installation

```bash
# Clone the repository
$ git clone https://github.com/sowmyapavani03/MENTAL-FITNESS-TRACKER-USING-SVM
$ cd MENTAL-FITNESS-TRACKER-USING-SVM

# Install dependencies
$ pip install -r requirements.txt

# Run the app
$ python app.py
```

Navigate to `http://localhost:5000` to access the web interface.

---

## 💡 Future Enhancements

* 📈 **Expand to Multi-Class Emotion Detection** (happy, sad, fear, etc.)
* 🌍 **Multilingual Input Support**
* 📱 **Mobile App Version**
* ☁️ **Cloud Deployment (AWS/GCP)**
* 🔐 **End-to-End Encryption & OAuth Login**

---


---

> "Artificial Intelligence won't replace therapists — but it can make support accessible to everyone."

📌 [Project Repo](https://github.com/sowmyapavani03/MENTAL-FITNESS-TRACKER-USING-SVM)
