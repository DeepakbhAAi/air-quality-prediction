"""
This file contains historical air quality data and weather impact analysis for Indian cities.
"""

import random
from datetime import datetime, timedelta
import numpy as np

# Generate synthetic historical data for the past year
def generate_historical_data(city_name, days=365):
    """
    Generate synthetic historical AQI data for a city.
    
    Args:
        city_name (str): Name of the city
        days (int): Number of days of historical data to generate
        
    Returns:
        dict: Dictionary with dates and AQI values
    """
    # Set random seed based on city name for consistent results
    random.seed(sum(ord(c) for c in city_name))
    
    # Base AQI values for different cities (higher for more polluted cities)
    base_aqi = {
        "Delhi": 160,
        "Mumbai": 90,
        "Kolkata": 120,
        "Chennai": 80,
        "Bangalore": 70,
        "Hyderabad": 85,
        "Ahmedabad": 100,
        "Pune": 80,
        "Jaipur": 110,
        "Lucknow": 130,
        "Kanpur": 140,
        "Nagpur": 90,
        "Patna": 130,
        "Indore": 100,
        "Thane": 85,
        "Bhopal": 90,
        "Visakhapatnam": 80,
        "Vadodara": 90,
        "Agra": 110,
        "Varanasi": 120
    }
    
    # Default base AQI if city not in the list
    city_base_aqi = base_aqi.get(city_name, 100)
    
    # Generate data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = {}
    current_date = start_date
    
    while current_date <= end_date:
        # Seasonal variation (higher in winter, lower in summer)
        month = current_date.month
        if 11 <= month <= 2:  # Winter
            seasonal_factor = 1.3
        elif 3 <= month <= 5:  # Spring
            seasonal_factor = 0.9
        elif 6 <= month <= 8:  # Summer/Monsoon
            seasonal_factor = 0.7
        else:  # Fall
            seasonal_factor = 1.1
        
        # Weekend variation (slightly lower on weekends)
        weekday = current_date.weekday()
        weekend_factor = 0.9 if weekday >= 5 else 1.0
        
        # Random daily variation
        daily_variation = random.uniform(0.8, 1.2)
        
        # Calculate AQI for the day
        daily_aqi = city_base_aqi * seasonal_factor * weekend_factor * daily_variation
        
        # Add some random spikes for realism
        if random.random() < 0.05:  # 5% chance of a spike
            daily_aqi *= random.uniform(1.3, 1.8)
        
        # Ensure AQI is within reasonable bounds
        daily_aqi = max(30, min(500, daily_aqi))
        
        # Store the data
        data[current_date.strftime('%Y-%m-%d')] = daily_aqi
        
        # Move to next day
        current_date += timedelta(days=1)
    
    return data

# Weather impact factors
WEATHER_IMPACT = {
    "Temperature": {
        "High (>30°C)": {
            "impact": "Increases ozone (O₃) formation and can lead to photochemical smog.",
            "effect": 1.2,
            "pollutants": ["O3"]
        },
        "Moderate (15-30°C)": {
            "impact": "Moderate conditions generally lead to average pollution levels.",
            "effect": 1.0,
            "pollutants": []
        },
        "Low (<15°C)": {
            "impact": "Cold temperatures can lead to temperature inversions, trapping pollutants near the ground.",
            "effect": 1.3,
            "pollutants": ["PM2.5", "PM10", "NO2"]
        }
    },
    "Humidity": {
        "High (>70%)": {
            "impact": "High humidity can increase the formation of secondary particles and make the air feel more polluted.",
            "effect": 1.1,
            "pollutants": ["PM2.5"]
        },
        "Moderate (40-70%)": {
            "impact": "Moderate humidity generally has minimal impact on air quality.",
            "effect": 1.0,
            "pollutants": []
        },
        "Low (<40%)": {
            "impact": "Low humidity can increase dust and particulate matter in the air.",
            "effect": 1.15,
            "pollutants": ["PM10"]
        }
    },
    "Wind Speed": {
        "High (>20 km/h)": {
            "impact": "Strong winds disperse pollutants and generally improve air quality.",
            "effect": 0.7,
            "pollutants": []
        },
        "Moderate (5-20 km/h)": {
            "impact": "Moderate winds provide some dispersion of pollutants.",
            "effect": 0.9,
            "pollutants": []
        },
        "Low (<5 km/h)": {
            "impact": "Low wind speeds allow pollutants to accumulate, worsening air quality.",
            "effect": 1.4,
            "pollutants": ["PM2.5", "PM10", "NO2", "SO2", "O3"]
        }
    },
    "Precipitation": {
        "Heavy Rain": {
            "impact": "Rain washes out particulate matter and other pollutants from the air.",
            "effect": 0.6,
            "pollutants": ["PM2.5", "PM10", "SO2"]
        },
        "Light Rain": {
            "impact": "Light rain provides some washing effect for pollutants.",
            "effect": 0.8,
            "pollutants": ["PM2.5", "PM10"]
        },
        "No Rain": {
            "impact": "Lack of precipitation allows pollutants to accumulate in the atmosphere.",
            "effect": 1.1,
            "pollutants": []
        }
    }
}

# Health recommendations based on AQI levels
HEALTH_RECOMMENDATIONS = {
    "Good (0-50)": {
        "General Population": "Air quality is satisfactory, and air pollution poses little or no risk.",
        "Sensitive Groups": "Enjoy outdoor activities.",
        "Outdoor Activities": "All outdoor activities are safe.",
        "Indoor Precautions": "No special precautions needed.",
        "Mask Recommendation": "No masks needed for air quality reasons."
    },
    "Moderate (51-100)": {
        "General Population": "Air quality is acceptable for most individuals.",
        "Sensitive Groups": "Unusually sensitive people should consider reducing prolonged or heavy exertion.",
        "Outdoor Activities": "Most outdoor activities are safe.",
        "Indoor Precautions": "No special precautions needed.",
        "Mask Recommendation": "Generally not needed, but sensitive individuals may consider wearing masks during prolonged outdoor exposure."
    },
    "Unhealthy for Sensitive Groups (101-150)": {
        "General Population": "Members of the general public may not be affected.",
        "Sensitive Groups": "Active children and adults, and people with respiratory disease, such as asthma, should limit prolonged outdoor exertion.",
        "Outdoor Activities": "Limit strenuous outdoor activities, especially during peak pollution hours.",
        "Indoor Precautions": "Keep windows closed during high pollution periods.",
        "Mask Recommendation": "N95 or PM2.5 masks recommended for sensitive groups during outdoor activities."
    },
    "Unhealthy (151-200)": {
        "General Population": "Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects.",
        "Sensitive Groups": "Active children and adults, and people with respiratory disease, such as asthma, should avoid prolonged outdoor exertion; everyone else should limit prolonged outdoor exertion.",
        "Outdoor Activities": "Avoid prolonged outdoor activities. Reschedule outdoor events.",
        "Indoor Precautions": "Keep windows and doors closed. Consider using air purifiers.",
        "Mask Recommendation": "N95 or PM2.5 masks recommended for everyone during outdoor activities."
    },
    "Very Unhealthy (201-300)": {
        "General Population": "Health warnings of emergency conditions. The entire population is more likely to be affected.",
        "Sensitive Groups": "Active children and adults, and people with respiratory disease, such as asthma, should avoid all outdoor exertion; everyone else should limit outdoor exertion.",
        "Outdoor Activities": "Avoid all outdoor physical activities. Stay indoors as much as possible.",
        "Indoor Precautions": "Keep windows and doors closed. Use air purifiers. Seal gaps in windows and doors if possible.",
        "Mask Recommendation": "N95 or PM2.5 masks essential for any outdoor exposure. Consider double masking."
    },
    "Hazardous (301+)": {
        "General Population": "Health alert: everyone may experience more serious health effects.",
        "Sensitive Groups": "Everyone should avoid all outdoor exertion.",
        "Outdoor Activities": "Avoid all outdoor activities. Stay indoors.",
        "Indoor Precautions": "Keep all windows and doors closed. Use air purifiers. Seal any gaps. Create a clean air room if possible.",
        "Mask Recommendation": "N95 or PM2.5 masks essential even for brief outdoor exposure. Consider double masking."
    }
}

# Generate forecast data for the next 7 days
def generate_forecast_data(city_name, current_aqi, days=7):
    """
    Generate synthetic forecast AQI data for a city.
    
    Args:
        city_name (str): Name of the city
        current_aqi (float): Current AQI value
        days (int): Number of days to forecast
        
    Returns:
        dict: Dictionary with dates and forecasted AQI values
    """
    # Set random seed based on city name for consistent results
    random.seed(sum(ord(c) for c in city_name) + datetime.now().day)
    
    # Generate data
    start_date = datetime.now() + timedelta(days=1)
    
    forecast = {}
    base_aqi = current_aqi
    
    for i in range(days):
        forecast_date = start_date + timedelta(days=i)
        
        # Add some random variation for the forecast
        daily_variation = random.uniform(0.85, 1.15)
        
        # Weekend adjustment
        weekday = forecast_date.weekday()
        weekend_factor = 0.9 if weekday >= 5 else 1.0
        
        # Calculate forecasted AQI
        forecasted_aqi = base_aqi * daily_variation * weekend_factor
        
        # Ensure AQI is within reasonable bounds
        forecasted_aqi = max(30, min(500, forecasted_aqi))
        
        # Store the data
        forecast[forecast_date.strftime('%Y-%m-%d')] = forecasted_aqi
        
        # Slightly adjust the base for the next day (trend continuation)
        base_aqi = forecasted_aqi * random.uniform(0.95, 1.05)
    
    return forecast
