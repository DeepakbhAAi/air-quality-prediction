"""
This file contains functions for creating interactive map visualizations of air quality data.
"""

# City coordinates (latitude, longitude) for major Indian cities
CITY_COORDINATES = {
    "Delhi": [28.6139, 77.2090],
    "Mumbai": [19.0760, 72.8777],
    "Kolkata": [22.5726, 88.3639],
    "Chennai": [13.0827, 80.2707],
    "Bangalore": [12.9716, 77.5946],
    "Hyderabad": [17.3850, 78.4867],
    "Ahmedabad": [23.0225, 72.5714],
    "Pune": [18.5204, 73.8567],
    "Jaipur": [26.9124, 75.7873],
    "Lucknow": [26.8467, 80.9462],
    "Kanpur": [26.4499, 80.3319],
    "Nagpur": [21.1458, 79.0882],
    "Patna": [25.5941, 85.1376],
    "Indore": [22.7196, 75.8577],
    "Thane": [19.2183, 72.9781],
    "Bhopal": [23.2599, 77.4126],
    "Visakhapatnam": [17.6868, 83.2185],
    "Vadodara": [22.3072, 73.1812],
    "Agra": [27.1767, 78.0081],
    "Varanasi": [25.3176, 82.9739]
}

def get_aqi_color(aqi_value):
    """
    Get color for AQI value.
    
    Args:
        aqi_value (float): AQI value
        
    Returns:
        str: Hex color code
    """
    if aqi_value <= 50:
        return "#00e400"  # Green
    elif aqi_value <= 100:
        return "#ffff00"  # Yellow
    elif aqi_value <= 150:
        return "#ff7e00"  # Orange
    elif aqi_value <= 200:
        return "#ff0000"  # Red
    elif aqi_value <= 300:
        return "#99004c"  # Purple
    else:
        return "#7e0023"  # Maroon

def generate_map_data(city_aqi_data):
    """
    Generate data for map visualization.
    
    Args:
        city_aqi_data (dict): Dictionary with city names as keys and AQI values as values
        
    Returns:
        list: List of dictionaries with map data
    """
    map_data = []
    
    for city, aqi in city_aqi_data.items():
        if city in CITY_COORDINATES:
            map_data.append({
                "name": city,
                "lat": CITY_COORDINATES[city][0],
                "lon": CITY_COORDINATES[city][1],
                "aqi": aqi,
                "color": get_aqi_color(aqi)
            })
    
    return map_data
