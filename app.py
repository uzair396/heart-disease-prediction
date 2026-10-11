from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# The model file sits in the same folder as this app.py
MODEL_PATH = Path(__file__).parent / "heart_disease_model.pkl"

# Same four features, same order as in the notebook
FEATURES = ["sex", "cp", "thalach", "exang"]

# Labels shown to the user -> numbers the model was trained on
SEX_OPTIONS = {"Female": 0, "Male": 1}
CP_OPTIONS = {
    "Typical angina": 1,
    "Atypical angina": 2,
    "Non-anginal pain": 3,
    "Asymptomatic (no chest pain symptoms)": 4,
}
EXANG_OPTIONS = {"No": 0, "Yes": 1}

# The heart-rate range the model saw during training
TRAIN_MIN_HR, TRAIN_MAX_HR = 71, 202

st.set_page_config(page_title="Heart Disease Prediction System", layout="centered")


@st.cache_resource
def load_model(path):
    """Load the saved pipeline once and reuse it."""
    return joblib.load(path)


st.title("Heart Disease Prediction System")
st.write(
    "This app uses a machine learning model trained on the UCI Heart Disease "
    "(Cleveland) dataset. Enter the four details below to get a prediction."
)

st.warning(
    "Educational project only. This is NOT a medical diagnosis tool and is not "
    "a substitute for professional medical advice, diagnosis or treatment. "
    "If you are worried about your health, please see a doctor."
)

# Load the model, and stop politely if something is wrong
try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error(
        "Model file 'heart_disease_model.pkl' was not found. Make sure it is in "
        "the same folder as app.py."
    )
    st.stop()
except Exception as error:
    st.error(f"The model could not be loaded: {error}")
    st.stop()

st.subheader("Patient details")

col1, col2 = st.columns(2)
with col1:
    sex_label = st.selectbox("Sex", list(SEX_OPTIONS.keys()))
    thalach = st.number_input(
        "Maximum heart rate achieved (beats per minute)",
        min_value=60, max_value=220, value=150, step=1,
    )
with col2:
    cp_label = st.selectbox("Chest pain type", list(CP_OPTIONS.keys()))
    exang_label = st.selectbox("Exercise-induced angina", list(EXANG_OPTIONS.keys()))

if st.button("Predict"):
    # Build one row with the SAME column names and order used in training
    patient = pd.DataFrame(
        [[
            float(SEX_OPTIONS[sex_label]),
            float(CP_OPTIONS[cp_label]),
            float(thalach),
            float(EXANG_OPTIONS[exang_label]),
        ]],
        columns=FEATURES,
    )

    try:
        prediction = int(model.predict(patient)[0])
        probabilities = model.predict_proba(patient)[0]
    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.stop()

    st.subheader("Result")

    if prediction == 1:
        st.error("The model predicts: heart disease is LIKELY.")
    else:
        st.success("The model predicts: heart disease is UNLIKELY.")

    left, right = st.columns(2)
    left.metric("Probability of no heart disease", f"{probabilities[0] * 100:.1f}%")
    right.metric("Probability of heart disease", f"{probabilities[1] * 100:.1f}%")
    st.progress(float(probabilities[1]))

    st.caption(
        "The probability is the model's estimate based on only four inputs. "
        "It is not a certainty and can be wrong."
    )

    if thalach < TRAIN_MIN_HR or thalach > TRAIN_MAX_HR:
        st.info(
            f"Note: the model was trained on heart rates between {TRAIN_MIN_HR} and "
            f"{TRAIN_MAX_HR} bpm. Your value is outside that range, so the result "
            "may be less reliable."
        )
