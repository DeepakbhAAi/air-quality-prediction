"""
API routes for the Air Quality Prediction application.
"""

from fastapi import APIRouter, Query
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from backend.utils.city_data import CITIES, AQI_CATEGORIES, SEASONAL_VARIATIONS, TIME_OF_DAY_VARIATIONS
from backend.utils.pollutant_info import POLLUTANT_INFO, DEFAULT_POLLUTANT_VALUES, PRESET_SCENARIOS, get_pollutant_level_description
from backend.utils.historical_data import generate_historical_data, WEATHER_IMPACT, HEALTH_RECOMMENDATIONS, generate_forecast_data
from backend.utils.map_visualization import CITY_COORDINATES, get_aqi_color, generate_map_data
from backend.utils.data_processor import (
    predict_aqi_for_city, predict_aqi_with_custom_values, get_city_pollutant_levels,
    predict_aqi_with_time_factors, get_current_season, get_current_time_of_day,
    compare_cities_aqi
)

router = APIRouter()

# Pydantic models for request/response
class PollutantLevels(BaseModel):
    PM2_5: float
    PM10: float
    NO2: float
    SO2: float
    O3: float

class AQIPrediction(BaseModel):
    aqi: float
    category: str

class CityPollutants(BaseModel):
    PM2_5: float
    PM10: float
    NO2: float
    SO2: float
    O3: float

class CityComparison(BaseModel):
    city: str
    aqi: float
    category: str

class HistoricalData(BaseModel):
    date: str
    aqi: float

class ForecastData(BaseModel):
    date: str
    aqi: float

class MapData(BaseModel):
    name: str
    lat: float
    lon: float
    aqi: float
    color: str

# Routes
@router.get("/cities", response_model=List[str])
async def get_cities():
    """Get list of available cities"""
    return CITIES

@router.get("/aqi-categories", response_model=Dict[str, str])
async def get_aqi_categories():
    """Get AQI categories and descriptions"""
    return AQI_CATEGORIES

@router.get("/seasonal-variations", response_model=Dict[str, float])
async def get_seasonal_variations():
    """Get seasonal variation factors"""
    return SEASONAL_VARIATIONS

@router.get("/time-of-day-variations", response_model=Dict[str, float])
async def get_time_of_day_variations():
    """Get time of day variation factors"""
    return TIME_OF_DAY_VARIATIONS

@router.get("/pollutant-info", response_model=Dict[str, Any])
async def get_pollutant_info():
    """Get information about pollutants"""
    return POLLUTANT_INFO

@router.get("/default-pollutant-values", response_model=Dict[str, float])
async def get_default_pollutant_values():
    """Get default values for pollutants"""
    return DEFAULT_POLLUTANT_VALUES

@router.get("/preset-scenarios", response_model=Dict[str, Any])
async def get_preset_scenarios():
    """Get preset pollution scenarios"""
    return PRESET_SCENARIOS

@router.get("/weather-impact", response_model=Dict[str, Any])
async def get_weather_impact():
    """Get weather impact data"""
    return WEATHER_IMPACT

@router.get("/health-recommendations", response_model=Dict[str, Any])
async def get_health_recommendations():
    """Get health recommendations based on AQI"""
    return HEALTH_RECOMMENDATIONS

@router.get("/city-coordinates", response_model=Dict[str, List[float]])
async def get_city_coordinates():
    """Get coordinates for cities"""
    return CITY_COORDINATES

@router.get("/city/{city_name}/aqi", response_model=AQIPrediction)
async def get_city_aqi(city_name: str):
    """Get AQI prediction for a city"""
    aqi, category = predict_aqi_for_city(city_name)
    return {"aqi": aqi, "category": category}

@router.get("/city/{city_name}/pollutants", response_model=CityPollutants)
async def get_pollutants_for_city(city_name: str):
    """Get pollutant levels for a city"""
    pollutants = get_city_pollutant_levels(city_name)
    return {
        "PM2_5": pollutants["PM2.5"],
        "PM10": pollutants["PM10"],
        "NO2": pollutants["NO2"],
        "SO2": pollutants["SO2"],
        "O3": pollutants["O3"]
    }

@router.post("/custom-aqi", response_model=AQIPrediction)
async def predict_custom_aqi(pollutants: PollutantLevels):
    """Predict AQI with custom pollutant values"""
    aqi, category = predict_aqi_with_custom_values(
        pollutants.PM2_5, pollutants.PM10, pollutants.NO2, pollutants.SO2, pollutants.O3
    )
    return {"aqi": aqi, "category": category}

@router.get("/city/{city_name}/time-prediction", response_model=AQIPrediction)
async def get_time_prediction(
    city_name: str, 
    season: Optional[str] = None, 
    time_of_day: Optional[str] = None
):
    """Get AQI prediction with time factors"""
    if not season:
        season = get_current_season()
    if not time_of_day:
        time_of_day = get_current_time_of_day()
    
    aqi, category = predict_aqi_with_time_factors(city_name, season, time_of_day)
    return {"aqi": aqi, "category": category}

@router.get("/compare-cities", response_model=List[CityComparison])
async def compare_cities(cities: List[str] = Query(...)):
    """Compare AQI for multiple cities"""
    results = compare_cities_aqi(cities)
    return [
        {"city": city, "aqi": aqi, "category": category}
        for city, (aqi, category) in results.items()
    ]

@router.get("/city/{city_name}/historical", response_model=List[HistoricalData])
async def get_historical_data(city_name: str, days: int = 365):
    """Get historical AQI data for a city"""
    data = generate_historical_data(city_name, days)
    return [{"date": date, "aqi": aqi} for date, aqi in data.items()]

@router.get("/city/{city_name}/forecast", response_model=List[ForecastData])
async def get_forecast_data(city_name: str, days: int = 7):
    """Get forecast AQI data for a city"""
    current_aqi, _ = predict_aqi_for_city(city_name)
    data = generate_forecast_data(city_name, current_aqi, days)
    return [{"date": date, "aqi": aqi} for date, aqi in data.items()]

@router.get("/map-data", response_model=List[MapData])
async def get_map_data(cities: List[str] = Query(...)):
    """Get map visualization data for cities"""
    city_aqi_data = {}
    for city in cities:
        aqi, _ = predict_aqi_for_city(city)
        city_aqi_data[city] = aqi
    
    return generate_map_data(city_aqi_data)
