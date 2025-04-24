"""
This file contains information about air pollutants, their typical ranges,
health effects, and sources.
"""

# Pollutant information with typical ranges, health effects, and sources
POLLUTANT_INFO = {
    "PM2.5": {
        "name": "Fine Particulate Matter (PM2.5)",
        "unit": "μg/m³",
        "min_range": 0.0,
        "max_range": 250.0,
        "good_range": (0, 12),
        "moderate_range": (12.1, 35.4),
        "unhealthy_range": (35.5, 150),
        "hazardous_range": (150.1, 250),
        "description": "Fine particulate matter with diameter less than 2.5 micrometers. These particles can penetrate deep into the lungs and even enter the bloodstream.",
        "health_effects": "Can cause respiratory and cardiovascular issues, aggravate asthma, decrease lung function, and increase risk of heart attacks.",
        "sources": "Vehicle emissions, power plants, industrial processes, wildfires, and residential wood burning."
    },
    "PM10": {
        "name": "Coarse Particulate Matter (PM10)",
        "unit": "μg/m³",
        "min_range": 0.0,
        "max_range": 430.0,
        "good_range": (0, 54),
        "moderate_range": (54.1, 154),
        "unhealthy_range": (154.1, 254),
        "hazardous_range": (254.1, 430),
        "description": "Particulate matter with diameter less than 10 micrometers. These particles can enter the respiratory system.",
        "health_effects": "Can cause respiratory issues, aggravate asthma, and decrease lung function.",
        "sources": "Dust from roads and construction sites, agricultural operations, industrial processes, and wildfires."
    },
    "NO2": {
        "name": "Nitrogen Dioxide (NO₂)",
        "unit": "μg/m³",
        "min_range": 0.0,
        "max_range": 200.0,
        "good_range": (0, 53),
        "moderate_range": (53.1, 100),
        "unhealthy_range": (100.1, 150),
        "hazardous_range": (150.1, 200),
        "description": "A reddish-brown gas with a pungent odor. It is a major air pollutant and a precursor to smog and acid rain.",
        "health_effects": "Can irritate the respiratory system, aggravate respiratory diseases, and contribute to the development of asthma.",
        "sources": "Vehicle emissions, power plants, and industrial processes."
    },
    "SO2": {
        "name": "Sulfur Dioxide (SO₂)",
        "unit": "μg/m³",
        "min_range": 0.0,
        "max_range": 100.0,
        "good_range": (0, 35),
        "moderate_range": (35.1, 75),
        "unhealthy_range": (75.1, 90),
        "hazardous_range": (90.1, 100),
        "description": "A colorless gas with a strong odor. It is a major air pollutant and a precursor to acid rain.",
        "health_effects": "Can irritate the respiratory system, aggravate asthma, and contribute to respiratory and cardiovascular diseases.",
        "sources": "Burning of fossil fuels, especially coal and oil, industrial processes, and volcanic eruptions."
    },
    "O3": {
        "name": "Ozone (O₃)",
        "unit": "μg/m³",
        "min_range": 0.0,
        "max_range": 200.0,
        "good_range": (0, 54),
        "moderate_range": (54.1, 124),
        "unhealthy_range": (124.1, 164),
        "hazardous_range": (164.1, 200),
        "description": "A gas composed of three oxygen atoms. While stratospheric ozone protects us from UV radiation, ground-level ozone is a harmful air pollutant.",
        "health_effects": "Can irritate the respiratory system, reduce lung function, aggravate asthma, and cause inflammation of lung tissue.",
        "sources": "Formed by chemical reactions between oxides of nitrogen (NOx) and volatile organic compounds (VOCs) in the presence of sunlight."
    }
}

# Default values for each pollutant (typical urban values)
DEFAULT_POLLUTANT_VALUES = {
    "PM2.5": 35.0,
    "PM10": 70.0,
    "NO2": 40.0,
    "SO2": 20.0,
    "O3": 60.0
}

# Preset scenarios for different pollution conditions
PRESET_SCENARIOS = {
    "Clean Air (Low Pollution)": {
        "PM2.5": 8.0,
        "PM10": 20.0,
        "NO2": 15.0,
        "SO2": 5.0,
        "O3": 30.0,
        "description": "Typical values for areas with very good air quality, such as rural areas or cities with strong environmental regulations."
    },
    "Moderate Urban Pollution": {
        "PM2.5": 35.0,
        "PM10": 70.0,
        "NO2": 40.0,
        "SO2": 20.0,
        "O3": 60.0,
        "description": "Typical values for urban areas with moderate pollution levels."
    },
    "Heavy Traffic Pollution": {
        "PM2.5": 60.0,
        "PM10": 120.0,
        "NO2": 80.0,
        "SO2": 30.0,
        "O3": 40.0,
        "description": "Pollution levels near busy roads or in areas with heavy traffic congestion."
    },
    "Industrial Area": {
        "PM2.5": 70.0,
        "PM10": 140.0,
        "NO2": 60.0,
        "SO2": 50.0,
        "O3": 70.0,
        "description": "Typical values for areas near industrial facilities or power plants."
    },
    "Severe Smog Episode": {
        "PM2.5": 150.0,
        "PM10": 250.0,
        "NO2": 120.0,
        "SO2": 60.0,
        "O3": 150.0,
        "description": "Extreme pollution levels during severe smog episodes or air pollution emergencies."
    }
}

def get_pollutant_level_description(pollutant, value):
    """
    Get a description of the pollutant level based on its value.

    Args:
        pollutant (str): Pollutant name (PM2.5, PM10, NO2, SO2, O3)
        value (float): Pollutant value

    Returns:
        tuple: (level_description, color)
    """
    info = POLLUTANT_INFO.get(pollutant)
    if not info:
        return "Unknown", "gray"

    if value <= info["good_range"][1]:
        return "Good", "green"
    elif value <= info["moderate_range"][1]:
        return "Moderate", "yellow"
    elif value <= info["unhealthy_range"][1]:
        return "Unhealthy", "orange"
    else:
        return "Hazardous", "red"
