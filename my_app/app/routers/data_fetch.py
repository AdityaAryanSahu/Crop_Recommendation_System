import pandas as pd
import numpy as np
from PIL import Image
import os
os.environ['METEOSTAT_CACHE_DIR'] = '/tmp/meteostat'
os.environ['HOME'] = '/tmp'
from meteostat import Point, Monthly, Daily, Stations
from datetime import datetime
import requests
import warnings

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError
import requests_cache

cache_dir = '/tmp/cache'


os.makedirs('/tmp/meteostat', exist_ok=True)


warnings.filterwarnings('ignore', message='X has feature names')

# Setup requests cache
os.makedirs(cache_dir, exist_ok=True)
requests_cache.install_cache(
    os.path.join(cache_dir, 'weather_cache'),
    backend='sqlite',
    expire_after=172800 
)

RETRY_EXCEPTIONS = (requests.Timeout, requests.ConnectionError, requests.HTTPError)

@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(RETRY_EXCEPTIONS)
)
def get_weather_details(lat, lon):
    """Get weather details from Meteostat and Visual Crossing API"""
    API_KEY = os.environ.get("VISUAL_CROSSING_API_KEY")
    avg_rain = 0.0  # Default fallback
    temp = 0.0      # Default fallback
    humidity = 0.0  # Default fallback
    
    # Try to get rainfall data from Meteostat
    try:
        station = nearby_stations(lat, lon)
        if station is None:
            raise ValueError("No nearby station found for rainfall data.")
        
        location = Point(station['latitude'], station['longitude'])
        start = datetime(2019, 1, 1)
        end = datetime.now()
        rain_data = Monthly(location, start, end).fetch()
        
        if not rain_data.empty and 'prcp' in rain_data.columns:
            avg_rain = rain_data['prcp'].mean()
            print(f"[INFO] Meteostat rainfall data retrieved: {avg_rain:.2f}mm")
        else:
            print("[WARNING] No rainfall data available from Meteostat")
            
    except Exception as e:
        print(f"[ERROR] Meteostat (rainfall) issue: {e}. Using avg_rain = 0.0")

    # Try to get temperature and humidity from Visual Crossing
    try:
        if not API_KEY:
            print("[WARNING] VISUAL_CROSSING_API_KEY not set")
            return temp, humidity, avg_rain
            
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat}%2C{lon}?unitGroup=metric&key={API_KEY}&contentType=json"
        response = requests.get(url, timeout=(5, 10))
        response.raise_for_status()
        data = response.json()
        today_data = data.get("days", [{}])[0]
        
        humidity = today_data.get("humidity", 0.0)
        temp = today_data.get("temp", 0.0)
        print(f"[INFO] Visual Crossing data: temp={temp}°C, humidity={humidity}%")
        
    except requests.RequestException as e:
        print(f"[ERROR] VisualCrossing API issue: {e}. Using temp/humidity = 0.0")
    except Exception as e:
        print(f"[ERROR] JSON parsing or other issue: {e}")
    
    return temp, humidity, avg_rain


def get_coordinates(city_name):
    """Get coordinates of a city using OpenStreetMap Nominatim API"""
    try: 
        url = f"https://nominatim.openstreetmap.org/search?city={city_name}&format=json"
        response = requests.get(url, headers={'User-Agent': 'crop-recommendation-app'})
        response.raise_for_status()
        data = response.json()
    
        if data:
            lat = float(data[0]['lat'])
            lon = float(data[0]['lon'])
            print(f"[INFO] Coordinates for {city_name}: {lat}, {lon}")
            return lat, lon
        else:
            print(f"[WARNING] No coordinates found for {city_name}")
            return None, None
            
    except Exception as e:
        print(f"[ERROR] Error getting coordinates: {e}")
        return None, None


def nearby_stations(lat, lon):
    """Find nearest Meteostat weather station"""
    try:
      #  stations.max_age = 0 
        stations = Stations().nearby(lat, lon)
        station = stations.fetch(1)
        
        
        if station.empty:
            raise ValueError("No nearby station found")
            
        station_data = station.iloc[0]
        print(f"[INFO] Found nearby station: {station_data.get('name', 'Unknown')} at {station_data['latitude']}, {station_data['longitude']}")
        return station_data
        
    except Exception as e:
        print(f"[ERROR] Error finding nearby station: {e}")
        return None
