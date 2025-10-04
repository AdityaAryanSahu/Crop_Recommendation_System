from meteostat import Stations
from user_interface_backend import *

lat, lon= get_coordinates("jharsuguda")
station= nearby_stations(lat, lon)
print(station)
location = Point(station['latitude'], station['longitude'])
start = datetime(2014, 1, 1)
end = datetime(2024,12,31)
   
data = Monthly(location, start, end).fetch()
print(len(data['prcp']))