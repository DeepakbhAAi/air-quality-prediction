import streamlit as st
import joblib
import numpy as np
from PIL import Image

# Load model
model = joblib.load('aqi_model.pkl')

# Streamlit app
st.title("Air Quality Index (AQI) Prediction")

# Pollutant inputs
st.subheader("Enter pollutant concentrations:")
PM2_5 = st.number_input("PM2.5 (μg/m³)", min_value=0.0, max_value=1000.0, step=1.0)
PM10 = st.number_input("PM10 (μg/m³)", min_value=0.0, max_value=1000.0, step=1.0)
NO = st.number_input("NO (μg/m³)", min_value=0.0, max_value=1000.0, step=1.0)
NO2 = st.number_input("NO2 (μg/m³)", min_value=0.0, max_value=1000.0, step=1.0)
NOx = st.number_input("NOx (μg/m³)", min_value=0.0, max_value=1000.0, step=1.0)
NH3 = st.number_input("NH3 (μg/m³)", min_value=0.0, max_value=1000.0, step=1.0)
CO = st.number_input("CO (mg/m³)", min_value=0.0, max_value=10.0, step=0.1)
SO2 = st.number_input("SO2 (μg/m³)", min_value=0.0, max_value=1000.0, step=1.0)
O3 = st.number_input("O3 (μg/m³)", min_value=0.0, max_value=1000.0, step=1.0)
Benzene = st.number_input("Benzene (μg/m³)", min_value=0.0, max_value=100.0, step=0.1)
Toluene = st.number_input("Toluene (μg/m³)", min_value=0.0, max_value=100.0, step=0.1)
Xylene = st.number_input("Xylene (μg/m³)", min_value=0.0, max_value=100.0, step=0.1)

# Predict button
if st.button("Predict AQI"):
    features = np.array([[PM2_5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene]])
    predicted_aqi = model.predict(features)[0]
    st.success(f"Predicted AQI is {predicted_aqi:.2f}")

    # Show chart
    image = Image.open("aqi_chart.webp")
    st.image(image, caption="AQI categories and pollutant concentration ranges", use_column_width=True)
