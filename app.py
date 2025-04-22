import streamlit as st
import joblib
import numpy as np
from PIL import Image

# Load model and encoder
model = joblib.load('aqi_model.pkl')
city_encoder = joblib.load('city_encoder.pkl')

# Streamlit app
st.title("Air Quality Index (AQI) Prediction")

# City selection (just for display, not used in model)
city_list = city_encoder.classes_
selected_city = st.selectbox("Select a city", city_list)

# Pollutant inputs (your original pollutants only)
st.subheader("Enter pollutant concentrations:")
PM2_5 = st.number_input("PM2.5")
PM10 = st.number_input("PM10")
NO = st.number_input("NO")
NO2 = st.number_input("NO2")
NOx = st.number_input("NOx")
NH3 = st.number_input("NH3")
CO = st.number_input("CO")
SO2 = st.number_input("SO2")
O3 = st.number_input("O3")
Benzene = st.number_input("Benzene")
Toluene = st.number_input("Toluene")
Xylene = st.number_input("Xylene")

# Predict button
if st.button("Predict AQI"):
    features = np.array([[PM2_5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene]])
    predicted_aqi = model.predict(features)[0]
    st.success(f"Predicted AQI for {selected_city} is {predicted_aqi:.2f}")

    # ✅ Show image below result
    image = Image.open("aqi_chart.webp")
    st.image(image, caption="AQI categories and pollutant concentration ranges", use_column_width=True)
