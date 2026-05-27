# app.py

import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re
import pandas as pd
import plotly.express as px
import gdown
import os

from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Mental Health Sentiment Monitoring",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------------------------
# Custom CSS Styling
# ---------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

h1 {
    color: #00D4FF;
    text-align: center;
}

h2 {
    color: #00FFA3;
}

.stButton>button {
    background-color: #00D4FF;
    color: black;
    font-size: 18px;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-weight: bold;
}

.stTextArea textarea {
    background-color: #1E1E1E;
    color: white;
    border-radius: 10px;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    margin-top: 20px;
}

.tip-box {
    padding: 15px;
    border-radius: 10px;
    background-color: white;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Download Model from Google Drive
# ---------------------------------------------------

MODEL_URL = "https://drive.google.com/uc?id=1kxze0K5v-Ha5Xjbpb2HprimdAsbek6HD"

MODEL_PATH = "mental_health_rnn_model.h5"

if not os.path.exists(MODEL_PATH):

    with st.spinner("Downloading AI Model..."):

        gdown.download(
            MODEL_URL,
            MODEL_PATH,
            quiet=False
        )

# ---------------------------------------------------
# Load Model and Files
# ---------------------------------------------------

model = tf.keras.models.load_model(
    MODEL_PATH
)

with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

# ---------------------------------------------------
# Parameters
# ---------------------------------------------------

max_length = 50

# ---------------------------------------------------
# Emotional Guidance Dictionary
# ---------------------------------------------------

guidance = {

    "Anxiety": {
        "message": "Take a short break and breathe slowly.",
        "tip": "Try meditation or talk with someone you trust."
    },

    "Depression": {
        "message": "You are stronger than your current thoughts.",
        "tip": "Go for a walk and stay connected with loved ones."
    },

    "Stress": {
        "message": "Your mind deserves rest.",
        "tip": "Listen to calming music or practice yoga."
    },

    "Normal": {
        "message": "You seem emotionally balanced today.",
        "tip": "Maintain healthy routines and positive habits."
    },

    "Suicidal": {
        "message": "Please seek immediate support from trusted people.",
        "tip": "Contact a mental health professional or support helpline."
    },

    "Bipolar": {
        "message": "Emotional fluctuations can be managed.",
        "tip": "Maintain proper sleep and consistent routines."
    },

    "Personality disorder": {
        "message": "Every emotion deserves understanding.",
        "tip": "Practice mindfulness and emotional journaling."
    }
}

# ---------------------------------------------------
# Preprocessing Function
# ---------------------------------------------------

def preprocess_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    sequence = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        sequence,
        maxlen=max_length,
        padding='post'
    )

    return padded

# ---------------------------------------------------
# Prediction Function
# ---------------------------------------------------

def predict_emotion(text):

    processed = preprocess_text(text)

    prediction = model.predict(processed)

    predicted_index = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    emotion = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    return emotion, confidence, prediction[0]

# ---------------------------------------------------
# SECTION 1 — Header
# ---------------------------------------------------

st.markdown("""
<h1>🧠 AI-Based Mental Health Sentiment Monitoring System</h1>
<h4 style='text-align:center; color:lightgray;'>
Emotion Detection using Simple Recurrent Neural Networks
</h4>
""", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------
# SECTION 2 — About the Project
# ---------------------------------------------------

st.header("📘 About the Project")

st.write("""
This project uses Artificial Intelligence and Natural Language Processing (NLP)
to detect emotions from user-written text.

### Importance of Emotional AI
Emotional AI helps systems understand human emotions and mental states
through language patterns and sentiment analysis.

### NLP Applications
- Mental health monitoring
- Chatbots
- Customer feedback analysis
- Emotion recognition systems

### Role of RNN in Sequence Learning
Simple Recurrent Neural Networks (RNNs) process text sequentially,
remembering previous words using hidden states to understand context.
""")

st.divider()

# ---------------------------------------------------
# SECTION 3 — User Input Area
# ---------------------------------------------------

st.header("✍️ Enter Your Thoughts")

sample_sentences = [
    "I feel lonely and stressed.",
    "Today is one of the happiest days of my life.",
    "Nobody understands my feelings.",
    "I feel calm and motivated today."
]

selected_sample = st.selectbox(
    "📌 Sample Sentence Suggestions",
    ["Select a sample sentence"] + sample_sentences
)

default_text = ""

if selected_sample != "Select a sample sentence":
    default_text = selected_sample

user_input = st.text_area(
    "Your Text",
    value=default_text,
    placeholder="Enter your thoughts or feelings here...",
    height=180
)

st.divider()

# ---------------------------------------------------
# SECTION 4 — Prediction Button
# ---------------------------------------------------

if st.button("🔍 Analyze Emotion"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:

        emotion, confidence, probabilities = predict_emotion(user_input)

        if confidence > 80:
            status = "High Confidence Prediction"
        elif confidence > 60:
            status = "Moderate Confidence Prediction"
        else:
            status = "Low Confidence Prediction"

        # ---------------------------------------------------
        # SECTION 5 — Prediction Output
        # ---------------------------------------------------

        st.header("📊 Prediction Output")

        st.markdown(f"""
        <div class="result-box">

        <h3>Emotion Detected: {emotion}</h3>

        <h4>Confidence: {confidence:.2f}%</h4>

        <h4>Emotional Status: {status}</h4>

        </div>
        """, unsafe_allow_html=True)

        # ---------------------------------------------------
        # SECTION 6 — Visualization Area
        # ---------------------------------------------------

        st.header("📈 Sentiment Confidence Visualization")

        labels = label_encoder.classes_

        prob_df = pd.DataFrame({
            "Emotion": labels,
            "Probability": probabilities
        })

        fig = px.bar(
            prob_df,
            x="Emotion",
            y="Probability",
            title="Emotion Probability Distribution",
            text_auto=True
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---------------------------------------------------
        # SECTION 7 — Emotional Guidance
        # ---------------------------------------------------

        st.header("💡 Emotional Wellness Guidance")

        emotion_key = emotion

        if emotion_key in guidance:

            st.markdown(f"""
            <div class="tip-box">

            <h4>🌟 Motivational Message</h4>
            <p>{guidance[emotion_key]['message']}</p>

            <h4>🧘 Wellness Tip</h4>
            <p>{guidance[emotion_key]['tip']}</p>

            </div>
            """, unsafe_allow_html=True)

        else:

            st.info(
                "Stay positive and maintain a healthy lifestyle."
            )

        # ---------------------------------------------------
        # Additional Features
        # ---------------------------------------------------

        st.header("📌 Additional Insights")

        word_count = len(user_input.split())

        st.write(f"📝 Word Count: {word_count}")

        sentiment_strength = confidence / 100

        progress_value = min(int(sentiment_strength * 100), 100)

        st.write("Emotion Confidence Meter")

        st.progress(progress_value)

        # Download Prediction Report

        report = f"""
Mental Health Sentiment Analysis Report

User Text:
{user_input}

Predicted Emotion:
{emotion}

Confidence:
{confidence:.2f}%

Status:
{status}
"""

        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name="emotion_report.txt",
            mime="text/plain"
        )

# ---------------------------------------------------
# Footer
# ---------------------------------------------------

st.divider()

st.markdown("""
<center>
sentiment analysis
</center>
""", unsafe_allow_html=True)