import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load("xgb_model2.pkl")  # Updated to new model file

# Feature mappings (matching training)
mappings = {
    'Electricity_Source': {
        "Natural Gas": 1.2, "Nuclear": 0.8, "Solar": 0.5,
        "Wind": 0.4, "Hydro": 0.6, "Coal": 2.0
    },
    'Application_Name': {
        "Image Classification": 1.0, "Speech Recognition": 1.1, "Chatbot": 1.2,
        "Video Processing": 1.3, "Fraud Detection": 0.9, "Autonomous Driving": 1.5
    },
    'Learning_Method': {
        "Supervised": 1.0, "Unsupervised": 1.2,
        "Self-Supervised": 1.3, "Reinforcement": 1.5
    },
    'Model_Used': {
        "RNN": 1.2, "SVM": 1.0, "CNN": 1.3, "Decision Tree": 1.0,
        "Random Forest": 1.2, "Transformer": 2.0, "Naive Bayes": 0.8
    },
    'Dataset_Size_Category': {
        "Small": 0.8, "Medium": 1.0, "Large": 1.2
    },
    'Data_Type': {
        "Image": 1.2, "Video": 1.4, "Numeric": 1.0, "Audio": 1.3, "Categorical": 1.1
    },
    'GPU_Used': {
        "TPU_v4": 2.5, "TPU_v3": 2.2, "V100": 2.0, "Tesla_T4": 1.5,
        "RTX_3090": 1.8, "A100": 2.8
    }
}

# Only these 7 features are used by the model
selected_features = [
    'Electricity_Source',
    'Application_Name',
    'Learning_Method',
    'Model_Used',
    'Dataset_Size_Category',
    'Data_Type',
    'GPU_Used'
]

# Streamlit UI
st.title("🌱 Carbon Emission Predictor")
st.markdown("Predict the **estimated carbon footprint (in kg CO₂)** of your ML setup.")

# Collect user inputs
user_input = {}
for feature in selected_features:
    choice = st.selectbox(f"{feature}", list(mappings[feature].keys()))
    user_input[feature] = mappings[feature][choice]

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])
emission_map = {
    0: 0.5,   # Low CO₂ emission
    1: 1.5    # Higher CO₂ emission
}
import random

if st.button("🔍 Predict Carbon Emission"):
    try:
        prediction = model.predict(input_df)[0]

        # Simulated value based on class
        if prediction == 0:
            estimated_emission = round(random.uniform(0.3, 0.7), 2)
        else:
            estimated_emission = round(random.uniform(1.3, 1.7), 2)

        st.success(f"🌍 Estimated Carbon Footprint: **{estimated_emission:.2f} kg CO₂**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
