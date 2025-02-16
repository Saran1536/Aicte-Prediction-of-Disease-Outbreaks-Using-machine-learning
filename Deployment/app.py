import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load datasets to get feature names
import pandas as pd

file_path = r"D:\Aicte\AICTE-Prediction-of-Disease-Outbreaks-main\Data\parkinsons.csv"
parkinsons_df = pd.read_csv(file_path).drop(columns=['name'])
heart_df = pd.read_csv(r"D:\Aicte\AICTE-Prediction-of-Disease-Outbreaks-main\Data\heart.csv")
diabetes_df = pd.read_csv("../Data/diabetes.csv")

parkinsons_features = list(parkinsons_df.columns[:-1])
heart_features = list(heart_df.columns[:-1])
diabetes_features = list(diabetes_df.columns[:-1])

# Load models and scalers
def load_model(filename):
    try:
        with open(filename, "rb") as f:
            model, scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error(f"Error: {filename} not found!")
        return None, None

parkinsons_model, parkinsons_scaler = load_model("../Models/parkinsons_model.pkl")
heart_model, heart_scaler = load_model("../Models/heart_model.pkl")
diabetes_model, diabetes_scaler = load_model("../Models/diabetes_model.pkl")

# Function to make predictions
def make_prediction(model, scaler, user_data, condition_name):
    if model is None or scaler is None:
        st.error(f"⚠️ Error: {condition_name} model or scaler not loaded!")
        return
    
    try:
        scaled_data = scaler.transform([user_data])
        prediction = model.predict(scaled_data)

        st.success(f"🩺 {condition_name} Detected!" if prediction[0] == 1 else f"✅ No {condition_name}")
    
    except Exception as e:
        st.error(f"Error predicting {condition_name}: {str(e)}")

# Streamlit UI
st.title("Health Prediction System")

# Sidebar for disease selection
disease = st.sidebar.selectbox("Select Disease", ["Parkinson's", "Heart Disease", "Diabetes"])

if disease == "Parkinson's":
    st.header("Parkinson's Disease Prediction")
    user_inputs = [st.number_input(feature, value=0.0) for feature in parkinsons_features]
    if st.button("Predict Parkinson's"):
        make_prediction(parkinsons_model, parkinsons_scaler, user_inputs, "Parkinson's Disease")

elif disease == "Heart Disease":
    st.header("Heart Disease Prediction")
    user_inputs = [st.number_input(feature, value=0.0) for feature in heart_features]
    if st.button("Predict Heart Disease"):
        make_prediction(heart_model, heart_scaler, user_inputs, "Heart Disease")

elif disease == "Diabetes":
    st.header("Diabetes Prediction")
    user_inputs = [st.number_input(feature, value=0.0) for feature in diabetes_features]
    if st.button("Predict Diabetes"):
        make_prediction(diabetes_model, diabetes_scaler, user_inputs, "Diabetes")
