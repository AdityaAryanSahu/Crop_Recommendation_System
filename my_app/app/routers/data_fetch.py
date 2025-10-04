import pandas as pd
import numpy as np
from PIL import Image
from meteostat import Point, Monthly, Daily, Stations
from datetime import datetime
import requests
import os


# this gets the historic data from meteostat and return avg rain
def get_weather_details(lat, lon):
    API_KEY = os.environ.get("VISUAL_CROSSING_API_KEY") # replace your api key
    avg_rain = 0.0 # Default fallback
    temp = 0.0     # Default fallback
    humidity = 0.0
    try:
        #print(f"lat:{lat} lon:{lon}")  # fro debugging
        station= nearby_stations(lat, lon)
        if station is None:
             raise ValueError("No nearby station found for rainfall data.")
        location = Point(station['latitude'], station['longitude'])
        start = datetime(2019, 1, 1)
        end = datetime.now()
        rain_data = Monthly(location, start, end).fetch()
        avg_rain = rain_data['prcp'].mean()
    except Exception as e:
        print(f"[ERROR] Meteostat (rainfall) issue: {e}. Using avg_rain = 0.0")

    try:
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat}%2C{lon}?unitGroup=metric&key={API_KEY}&contentType=json"
        response = requests.get(url)
        response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
        data = response.json()
        today_data = data.get("days", [{}])[0]
              
        humidity = today_data.get("humidity", 0.0)
        temp = today_data.get("temp", 0.0)
    except requests.RequestException as e:
        print(f"[ERROR] VisualCrossing API issue: {e}. Using temp/humidity = 0.0")
    except Exception as e:
        print(f"[ERROR] JSON parsing or other issue: {e}")
    
    
    return temp, humidity, avg_rain

# this to get coordinates of a city for meteostat
def get_coordinates(city_name):
    try: 
        url = f"https://nominatim.openstreetmap.org/search?city={city_name}&format=json"
        response = requests.get(url, headers={'User-Agent': 'crop-recommendation-app'})
        data = response.json()
    
        if data:
            lat = float(data[0]['lat'])
            lon = float(data[0]['lon'])
            return lat, lon
        else:
            return None, None
    except Exception as e:
        print("[ERROR] error in coordinates")
   
# basically if any city is directly not in database, 
# nearest city/station is taken for smooth operation 
def nearby_stations(lat, lon):
    try:
        stations = Stations().nearby(lat, lon)
        station = stations.fetch(1)
        if station.empty:
            raise ValueError("No nearby station found")
        return station.iloc[0]
    except Exception  as e:
        print(f"[ERROR] Error finding nearby station: {e}")
