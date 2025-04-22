import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('aqi_model.pkl')

# Streamlit UI
st.title("Air Quality Prediction")

st.write("Enter pollutant levels below:")

pm2_5 = st.number_input("PM2.5", min_value=0.0)
pm10 = st.number_input("PM10", min_value=0.0)
no2 = st.number_input("NO2", min_value=0.0)
so2 = st.number_input("SO2", min_value=0.0)
o3 = st.number_input("O3", min_value=0.0)

if st.button("Predict AQI"):
    features = np.array([[pm2_5, pm10, no2, so2, o3]])  # Exactly 5 features
    predicted_aqi = model.predict(features)[0]
    st.success(f"Predicted AQI: {predicted_aqi:.2f}")
    
    # Show image
    st.image("chart.webp", caption="AQI Levels", use_column_width=True)
