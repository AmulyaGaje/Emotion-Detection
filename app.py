# app.py

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import re
import os
import gdown

import plotly.express as px
import plotly.graph_objects as go

from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Mental Health Dashboard",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #111827,
        #1e293b
    );
    color: white;
}

.main-title {
    text-align: center;
    font-size: 52px;
    font-weight: 700;
    color: #38bdf8;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 18px;
    margin-bottom: 20px;
}

.glass {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 25px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    margin-bottom: 20px;
}

.metric-card {
    background: linear-gradient(
        135deg,
        rgba(56,189,248,0.25),
        rgba(59,130,246,0.15)
    );
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.1);
}

.metric-value {
    font-size: 32px;
    font-weight: bold;
    color: #38bdf8;
}

.metric-label {
    color: #e2e8f0;
    font-size: 16px;
}

.stButton>button {
    background: linear-gradient(
        90deg,
        #06b6d4,
        #3b82f6
    );
    color: white;
    border: none;
    border-radius: 15px;
    height: 55px;
    font-size: 20px;
    font-weight: 600;
    width: 100%;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.02);
}

textarea {
    background-color: rgba(255,255,255,0.08) !important;
    color: white !important;
    border-radius: 15px !important;
}

.result-box {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 18px;
    margin-top: 20px;
}

.tip-box {
    background: rgba(56,189,248,0.1);
    padding: 20px;
    border-radius: 15px;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🧭 Dashboard Navigation")

menu = st.sidebar.radio(
    "Select Section",
    [
        "Dashboard",
        "Prediction",
        "Analytics",
        "Wellness Tips"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info("""
This AI dashboard detects emotional sentiment
using NLP and Recurrent Neural Networks.
""")

# =========================================================
# DOWNLOAD MODEL
# =========================================================

MODEL_URL = "https://drive.google.com/uc?id=1kxze0K5v-Ha5Xjbpb2HprimdAsbek6HD"

MODEL_PATH = "mental_health_rnn_model.h5"

if not os.path.exists(MODEL_PATH):

    with st.spinner("Downloading AI Model..."):

        gdown.download(
            MODEL_URL,
            MODEL_PATH,
            quiet=False
        )

# =========================================================
# LOAD MODEL & FILES
# =========================================================

model = tf.keras.models.load_model(
    MODEL_PATH
)

with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

# =========================================================
# PARAMETERS
# =========================================================

max_length = 50

# =========================================================
# EMOTIONAL GUIDANCE
# =========================================================

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
        "tip": "Contact a mental health professional immediately."
    },

    "Bipolar": {
        "message": "Emotional fluctuations can be managed.",
        "tip": "Maintain healthy sleep and routines."
    },

    "Personality disorder": {
        "message": "Every emotion deserves understanding.",
        "tip": "Practice mindfulness and journaling."
    }
}

# =========================================================
# PREPROCESSING
# =========================================================

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

# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_emotion(text):

    processed = preprocess_text(text)

    prediction = model.predict(processed)

    predicted_index = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    emotion = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    return emotion, confidence, prediction[0]

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="glass">

<h1 class="main-title">
🧠 AI-Based Mental Health Sentiment Monitoring System
</h1>

<p class="subtitle">
Emotion Detection using Simple Recurrent Neural Networks
</p>

</div>
""", unsafe_allow_html=True)

# =========================================================
# ABOUT PROJECT
# =========================================================

with st.container():

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.header("📘 About the Project")

    st.write("""
This project uses Artificial Intelligence and Natural Language Processing (NLP)
to detect emotions from user-written text.

### Importance of Emotional AI
Emotional AI helps systems understand human emotions and mental states.

### NLP Applications
- Mental Health Monitoring
- Chatbots
- Customer Feedback Analysis
- Emotion Recognition

### Role of RNN
RNN remembers previous words using hidden states
to understand sequence and context.
""")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# INPUT SECTION
# =========================================================

st.markdown('<div class="glass">', unsafe_allow_html=True)

st.header("✍️ Enter Your Thoughts")

sample_sentences = [
    "I feel lonely and stressed.",
    "Nobody understands my feelings.",
    "I feel calm and motivated today.",
    "Today is one of the happiest days of my life."
]

selected_sample = st.selectbox(
    "📌 Sample Sentences",
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

st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# ANALYZE BUTTON
# =========================================================

if st.button("🔍 Analyze Emotion"):

    if user_input.strip() == "":

        st.warning("Please enter some text.")

    else:

        emotion, confidence, probabilities = predict_emotion(user_input)

        # =================================================
        # STATUS
        # =================================================

        if confidence > 80:
            status = "High Confidence"

        elif confidence > 60:
            status = "Moderate Confidence"

        else:
            status = "Low Confidence"

        # =================================================
        # METRICS
        # =================================================

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">
                    {emotion}
                </div>
                <div class="metric-label">
                    Emotion Detected
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">
                    {confidence:.2f}%
                </div>
                <div class="metric-label">
                    Confidence Score
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:

            risk = "Low"

            if emotion in ["Depression", "Suicidal"]:
                risk = "High"

            elif emotion in ["Stress", "Anxiety"]:
                risk = "Moderate"

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">
                    {risk}
                </div>
                <div class="metric-label">
                    Emotional Risk
                </div>
            </div>
            """, unsafe_allow_html=True)

        # =================================================
        # PREDICTION OUTPUT
        # =================================================

        st.markdown("""
        <br>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-box">

        <h2>📊 Prediction Result</h2>

        <h3>Emotion Detected: {emotion}</h3>

        <h3>Confidence: {confidence:.2f}%</h3>

        <h3>Status: {status}</h3>

        </div>
        """, unsafe_allow_html=True)

        # =================================================
        # PROBABILITY CHART
        # =================================================

        st.markdown("""
        <br>
        """, unsafe_allow_html=True)

        labels = label_encoder.classes_

        prob_df = pd.DataFrame({
            "Emotion": labels,
            "Probability": probabilities
        })

        fig = px.bar(
            prob_df,
            x="Emotion",
            y="Probability",
            text_auto=True,
            title="Emotion Probability Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # =================================================
        # GAUGE CHART
        # =================================================

        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            title={'text': "Confidence Meter"},
            gauge={
                'axis': {'range': [0,100]},
                'bar': {'color': "#38bdf8"},
                'steps': [
                    {'range': [0,50], 'color': "#1e293b"},
                    {'range': [50,80], 'color': "#334155"},
                    {'range': [80,100], 'color': "#0ea5e9"}
                ]
            }
        ))

        st.plotly_chart(
            gauge_fig,
            use_container_width=True
        )

        # =================================================
        # WELLNESS SCORE
        # =================================================

        wellness_score = 100 - confidence

        if emotion == "Normal":
            wellness_score = 95

        st.subheader("🌿 Wellness Score")

        st.progress(int(wellness_score))

        st.write(
            f"Overall Wellness Score: {wellness_score:.1f}/100"
        )

        # =================================================
        # WELLNESS GUIDANCE
        # =================================================

        st.subheader("💡 Emotional Wellness Guidance")

        if emotion in guidance:

            st.markdown(f"""
            <div class="tip-box">

            <h4>🌟 Motivational Message</h4>
            <p>{guidance[emotion]['message']}</p>

            <h4>🧘 Wellness Tip</h4>
            <p>{guidance[emotion]['tip']}</p>

            </div>
            """, unsafe_allow_html=True)

        # =================================================
        # TEXT ANALYTICS
        # =================================================

        st.subheader("⌨️ Text Analytics")

        word_count = len(user_input.split())

        char_count = len(user_input)

        avg_word_length = sum(
            len(word)
            for word in user_input.split()
        ) / max(word_count,1)

        col1, col2, col3 = st.columns(3)

        col1.metric("Words", word_count)

        col2.metric("Characters", char_count)

        col3.metric(
            "Avg Word Length",
            round(avg_word_length,2)
        )

        # =================================================
        # SESSION HISTORY
        # =================================================

        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            "Emotion": emotion,
            "Confidence": confidence
        })

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.subheader("📈 Emotion History")

        st.dataframe(
            history_df,
            use_container_width=True
        )

        # =================================================
        # DOWNLOAD REPORT
        # =================================================

        report = f"""
AI Mental Health Monitoring Report

User Text:
{user_input}

Emotion Detected:
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

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<hr>

<center>

<h4>
🚀 Developed using Streamlit, TensorFlow, NLP & RNN
</h4>

<p>
AI-Based Mental Health Monitoring Dashboard
</p>

</center>
""", unsafe_allow_html=True)
