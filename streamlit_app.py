import os
from pathlib import Path

import joblib
import numpy as np
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"


@st.cache_resource
def load_local_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def predict_locally(study_hours: float, prev_score: float):
    model = load_local_model()
    input_data = np.array([[study_hours, prev_score]])
    prediction = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]
    confidence = round(float(probabilities[prediction]), 3)
    return {
        "prediction": prediction,
        "result_label": "Pass" if prediction == 1 else "Fail",
        "confidence": confidence,
    }


st.set_page_config(
    page_title="Student Predictor",
    page_icon="🎓",
    layout="centered",
)

st.title("🎓 Student Pass / Fail Predictor")
st.markdown("Enter the student's details below to predict their result.")
st.markdown("---")

# Sliders for input
col1, col2 = st.columns(2)
with col1:
    study_hours = st.slider(
        label="Study Hours Per Day",
        min_value=0.0,
        max_value=10.0,
        value=4.0,
        step=0.5,
    )
with col2:
    prev_score = st.slider(
        label="Previous Exam Score",
        min_value=0.0,
        max_value=100.0,
        value=60.0,
        step=1.0,
    )

st.markdown("---")


# Predict Button
if st.button("🔍 Predict Result", use_container_width=True):
    result = None

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"study_hours": study_hours, "prev_score": prev_score},
            timeout=5,
        )
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.RequestException:
        st.warning(
            "FastAPI backend is not running. Falling back to the local model in Streamlit. "
            f"API URL attempted: {API_URL}"
        )
        try:
            result = predict_locally(study_hours, prev_score)
        except Exception as exc:
            st.error(f"Prediction failed locally as well: {exc}")
            result = None

    if result is not None:
        if result["prediction"] == 1:
            st.success(f"🎉 Result: **{result['result_label']}**")
        else:
            st.error(f"⚠️ Result: **{result['result_label']}**")

        st.metric(
            label="Model Confidence",
            value=f"{result['confidence'] * 100:.1f}%",
        )

        with st.expander("See raw API response"):
            st.json(result)

# Live input summary
st.markdown("---")
st.subheader("Current Input Summary")
m1, m2 = st.columns(2)
m1.metric("Study Hours", f"{study_hours} hrs/day")
m2.metric("Previous Score", f"{prev_score}/100")
