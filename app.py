import streamlit as st 
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model  # ✅ Import fixed
import base64
def set_image_local(image_path):
    with open(image_path, "rb") as file:
        img = file.read()
    base64_image = base64.b64encode(img).decode("utf-8")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{base64_image}");
            background-size: cover;
            background-repeat: no-repeat;
            #background-position: center;
            #background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_image_local(r"img1.jpg")
# Load the trained model
model = load_model(r"detection_model.h5")  

# Function to make predictions
def predict_fraud(data):
    prediction = model.predict(np.array([data]))
    return "Fraudulent Transaction" if prediction[0][0] > 0.5 else "Genuine Transaction"

# Streamlit UI
st.title("🔍 Fraud Detection System")
st.write("Enter transaction details to predict if it's fraudulent or genuine.")

# Input fields (Modify based on your dataset features)
features = []
num_features = 30  # Change this based on the number of features

for i in range(num_features):
    value = st.number_input(f"v {i+1}", value=0.0)
    features.append(value)

# ✅ Fixed: Missing label for number input
Amount = st.number_input("Transaction Amount", value=100, min_value=100, max_value=900000)

# Predict button
if st.button("Predict"):  
    result = predict_fraud(features)
    st.subheader(f"Result: {result}")

