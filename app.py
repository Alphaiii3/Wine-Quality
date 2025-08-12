# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load files
@st.cache_data
def load_data():
    return pd.read_csv("data/winequality-dataset_updated.csv")

@st.cache_resource
def load_model():
    return pickle.load(open("newmodel (1).pkl", "rb"))

@st.cache_resource
def load_scaler():
    return pickle.load(open("scaler.pkl", "rb"))

df = load_data()
model = load_model()
scaler = load_scaler()

features = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
    'density', 'pH', 'sulphates', 'alcohol'
]

# Navigation
menu = st.sidebar.radio("Menu", ["Prediction", "Model Performance"])

# Prediction
if menu == "Prediction":
    st.header("🍷 Wine Quality Prediction")

    inputs = []
    for f in features:
        val = st.number_input(f"Enter {f}", value=float(df[f].median()))
        inputs.append(val)

    if st.button("Predict"):
        X_user = np.array(inputs).reshape(1, -1)
        X_user_s = scaler.transform(X_user)

        prob_good = model.predict_proba(X_user_s)[0, 1]
        pred = "Good" if prob_good >= 0.5 else "Not Good"

        st.success(f"Prediction: {pred}")
        st.info(f"Confidence: {prob_good:.2f}")

# Performance
elif menu == "Model Performance":
    st.header("📊 Model Performance")

    X = df[features]
    y = (df['quality'] >= 7).astype(int)
    X_s = scaler.transform(X)
    y_pred = model.predict(X_s)

    acc = accuracy_score(y, y_pred)
    st.write(f"Accuracy: {acc:.2f}")
    st.text(classification_report(y, y_pred))

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Not Good', 'Good'], yticklabels=['Not Good', 'Good'], ax=ax)
    st.pyplot(fig)
