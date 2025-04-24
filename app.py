"""
Streamlit frontend for the Air Quality Prediction application.
"""

import streamlit as st
import uuid

# Set page configuration
st.set_page_config(
    page_title="Air Quality Prediction for Indian Cities",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .card {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    
    .stButton button:hover {
        background-color: #45a049;
    }
    
    h1, h2, h3 {
        color: #2c3e50;
    }
    
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
    
    .loader {
        border: 16px solid #f3f3f3;
        border-radius: 50%;
        border-top: 16px solid #3498db;
        width: 120px;
        height: 120px;
        animation: spin 2s linear infinite;
        margin: auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# Import components
from frontend.src.components import CityPrediction, TimePrediction, CityComparison, CustomPrediction

# Streamlit UI
st.title("Air Quality Prediction for Indian Cities")

# Create tabs for different prediction methods
tab1, tab2, tab3, tab4 = st.tabs([
    "City Prediction",
    "Time Prediction",
    "City Comparison",
    "Custom Prediction"
])

# Tab 1: City-based prediction
with tab1:
    CityPrediction.render()

# Tab 2: Time-based prediction
with tab2:
    TimePrediction.render()

# Tab 3: City comparison
with tab3:
    CityComparison.render()

# Tab 4: Custom prediction
with tab4:
    CustomPrediction.render()
