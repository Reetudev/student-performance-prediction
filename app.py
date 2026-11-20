import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("student_prediction_model.pkl")

st.title("🎓 Student Performance Prediction System")

st.write("Enter student details to predict final grade.")

# Inputs
age = st.slider("Age", 15, 22, 18)
studytime = st.slider("Study Time (1-4)", 1, 4, 2)
failures = st.slider("Previous Failures", 0, 4, 0)
absences = st.slider("Absences", 0, 30, 5)

g1 = st.slider("G1 Grade", 0, 20, 10)
g2 = st.slider("G2 Grade", 0, 20, 10)

if st.button("Predict Final Grade"):

    data = {
        "age": age,
        "studytime": studytime,
        "failures": failures,
        "absences": absences,
        "G1": g1,
        "G2": g2
    }

    df = pd.DataFrame([data])

    # Add missing columns expected by model
    model_features = model.feature_names_in_

    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    df = df[model_features]

    prediction = model.predict(df)

    st.success(f"Predicted Final Grade (G3): {prediction[0]:.2f}")
