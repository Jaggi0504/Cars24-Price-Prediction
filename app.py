import streamlit as st
import joblib
import pandas as pd

st.markdown(
    """
    <style>
    body {
        background-color: #f4f4f4;
    }
    .stApp {
        background-color: #06659d;
    }
    .title-container {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: white;
    }
    .stTextInput, .stSelectbox, .stNumberInput {
        border-radius: 10px;
        padding: 5px;
        background-color: black;
    }
    .stButton>button {
        background-color: #007BFF;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 18px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title-container">Car Price Prediction App</div>', unsafe_allow_html=True)

# Load model and encoders
model = joblib.load("random_forest_model.pkl")
model_name_encoder = joblib.load("le_model_name.pkl")
transmission_encoder = joblib.load("le_transmission.pkl")
fuel_type_encoder = joblib.load("le_fuel_type.pkl")
spare_key_encoder = joblib.load("le_spare_key.pkl")

st.write("Want to predict the price of a car leveraging Machine Learning?")
st.write("Currently, the model is restricted to predicting the price of specific Maruti models. Please find out by selecting from the dropdown.")

# Dropdown options (assuming these were used during encoding)
model_names = model_name_encoder.classes_
transmission_types = transmission_encoder.classes_
fuel_types = fuel_type_encoder.classes_
spare_keys = spare_key_encoder.classes_

# User Inputs
model_name = st.selectbox("Select Model Name", model_names)
manufacturing_year = st.number_input("Enter Manufacturing Year", min_value=2010, max_value=2023, step=1)
engine_capacity = st.number_input("Enter Engine Capacity (cc)", min_value=800, max_value=1450, step=100)
spare_key = st.selectbox("Does the car have a spare key?", spare_keys)
transmission = st.selectbox("Select Transmission Type", transmission_types)
km_driven = st.number_input("Enter Kilometers Driven", min_value=0, max_value=500000, step=500)
ownership = st.number_input("Enter Ownership (1st, 2nd, etc.)", min_value=1, max_value=5, step=1)
fuel_type = st.selectbox("Select Fuel Type", fuel_types)
imperfections = st.number_input("Enter Number of Imperfections", min_value=0, max_value=50, step=1)
repainted_parts = st.number_input("Enter Number of Repainted Parts", min_value=0, max_value=27, step=1)

# Convert categorical values using Label Encoders
model_name_encoded = model_name_encoder.transform([model_name])[0]
transmission_encoded = transmission_encoder.transform([transmission])[0]
fuel_type_encoded = fuel_type_encoder.transform([fuel_type])[0]
spare_key_encoded = spare_key_encoder.transform([spare_key])[0]

# Make prediction
if st.button("Predict"):
    input_data = pd.DataFrame([[model_name_encoded, manufacturing_year, engine_capacity, 
                                spare_key_encoded, transmission_encoded, km_driven, 
                                ownership, fuel_type_encoded, imperfections, repainted_parts]], 
                              columns=["Model Name", "Manufacturing_year", "Engine capacity", 
                                       "Spare key", "Transmission", "KM driven", 
                                       "Ownership", "Fuel type", "Imperfections", "Repainted Parts"])
    
    prediction = model.predict(input_data)[0]
    st.write(f"Predicted Price: ₹{prediction:,.2f}")