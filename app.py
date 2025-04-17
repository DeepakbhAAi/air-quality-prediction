
import streamlit as st
import joblib
import numpy as np

# Load model and encoder
model = joblib.load('aqi_model.pkl')
city_encoder = joblib.load('city_encoder.pkl')

# Streamlit App UI
st.title("🌫️ Air Quality Index (AQI) Prediction")
st.markdown("Enter the details below to predict AQI for a city.")

# Input fields
city = st.selectbox("Select City", city_encoder.classes_)
pm25 = st.number_input("PM2.5", min_value=0.0)
pm10 = st.number_input("PM10", min_value=0.0)
no2 = st.number_input("NO2", min_value=0.0)
co = st.number_input("CO", min_value=0.0)
o3 = st.number_input("O3", min_value=0.0)

if st.button("Predict AQI"):
    city_encoded = city_encoder.transform([city])[0]
    features = np.array([[city_encoded, pm25, pm10, no2, co, o3]])
    predicted_aqi = model.predict(features)[0]
    st.success(f"🌍 Predicted AQI for {city}: **{round(predicted_aqi, 2)}**")
