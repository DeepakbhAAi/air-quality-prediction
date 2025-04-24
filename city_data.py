"""
This file contains data about major Indian cities for air quality prediction.
"""

# List of major Indian cities
CITIES = [
    "Delhi",
    "Mumbai",
    "Kolkata",
    "Chennai",
    "Bangalore",
    "Hyderabad",
    "Ahmedabad",
    "Pune",
    "Jaipur",
    "Lucknow",
    "Kanpur",
    "Nagpur",
    "Patna",
    "Indore",
    "Thane",
    "Bhopal",
    "Visakhapatnam",
    "Vadodara",
    "Agra",
    "Varanasi"
]

# Default feature values for each city (PM2.5, PM10, NO2, SO2, O3)
# These are placeholder values and should be replaced with actual historical averages
CITY_DEFAULT_FEATURES = {
    "Delhi": [90.0, 180.0, 80.0, 20.0, 50.0],
    "Mumbai": [60.0, 120.0, 60.0, 15.0, 40.0],
    "Kolkata": [70.0, 140.0, 65.0, 18.0, 45.0],
    "Chennai": [50.0, 100.0, 50.0, 12.0, 35.0],
    "Bangalore": [45.0, 90.0, 45.0, 10.0, 30.0],
    "Hyderabad": [55.0, 110.0, 55.0, 14.0, 38.0],
    "Ahmedabad": [65.0, 130.0, 60.0, 16.0, 42.0],
    "Pune": [50.0, 100.0, 50.0, 12.0, 35.0],
    "Jaipur": [75.0, 150.0, 70.0, 19.0, 48.0],
    "Lucknow": [80.0, 160.0, 75.0, 20.0, 49.0],
    "Kanpur": [85.0, 170.0, 78.0, 21.0, 52.0],
    "Nagpur": [60.0, 120.0, 60.0, 15.0, 40.0],
    "Patna": [80.0, 160.0, 75.0, 20.0, 49.0],
    "Indore": [65.0, 130.0, 60.0, 16.0, 42.0],
    "Thane": [55.0, 110.0, 55.0, 14.0, 38.0],
    "Bhopal": [60.0, 120.0, 60.0, 15.0, 40.0],
    "Visakhapatnam": [50.0, 100.0, 50.0, 12.0, 35.0],
    "Vadodara": [60.0, 120.0, 60.0, 15.0, 40.0],
    "Agra": [75.0, 150.0, 70.0, 19.0, 48.0],
    "Varanasi": [70.0, 140.0, 65.0, 18.0, 45.0]
}

# AQI categories and their corresponding health implications
AQI_CATEGORIES = {
    "Good (0-50)": "Air quality is considered satisfactory, and air pollution poses little or no risk.",
    "Moderate (51-100)": "Air quality is acceptable; however, for some pollutants, there may be a moderate health concern for a very small number of people.",
    "Unhealthy for Sensitive Groups (101-150)": "Members of sensitive groups may experience health effects. The general public is not likely to be affected.",
    "Unhealthy (151-200)": "Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects.",
    "Very Unhealthy (201-300)": "Health warnings of emergency conditions. The entire population is more likely to be affected.",
    "Hazardous (301+)": "Health alert: everyone may experience more serious health effects."
}

# Seasonal variations in air quality (multipliers for different seasons)
SEASONAL_VARIATIONS = {
    "Winter (Dec-Feb)": 1.3,  # Higher pollution in winter
    "Summer (Mar-Jun)": 0.8,  # Lower pollution in summer
    "Monsoon (Jul-Sep)": 0.6,  # Lowest pollution during monsoon
    "Post-Monsoon (Oct-Nov)": 1.1  # Moderate pollution in post-monsoon
}

# Time of day variations in air quality (multipliers for different times)
TIME_OF_DAY_VARIATIONS = {
    "Morning (6-9 AM)": 1.2,  # Higher pollution during morning rush hour
    "Midday (10 AM-4 PM)": 0.9,  # Moderate pollution during midday
    "Evening (5-8 PM)": 1.3,  # Higher pollution during evening rush hour
    "Night (9 PM-5 AM)": 0.8  # Lower pollution during night
}

def get_aqi_category(aqi_value):
    """
    Return the AQI category based on the AQI value.
    """
    if aqi_value <= 50:
        return "Good (0-50)"
    elif aqi_value <= 100:
        return "Moderate (51-100)"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups (101-150)"
    elif aqi_value <= 200:
        return "Unhealthy (151-200)"
    elif aqi_value <= 300:
        return "Very Unhealthy (201-300)"
    else:
        return "Hazardous (301+)"
