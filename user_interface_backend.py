import pandas as pd
import numpy as np
import joblib
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from meteostat import Point, Monthly, Daily, Stations
from datetime import datetime
import requests

model=joblib.load(r"D:\Crop_rec_system\voting_clf.pkl")
mms=joblib.load(r"D:\Crop_rec_system\sc.pkl")
selector=joblib.load(r"D:\Crop_rec_system\selector.pkl")
encoder=joblib.load(r'D:\Crop_rec_system\label_encoder.pkl')


class SoilCNN(nn.Module):
    def __init__(self, num_classes=5, num_dense=64, dropout=0.2):
        super(SoilCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, num_dense),
            nn.ReLU(),
            nn.BatchNorm1d(num_dense),
            nn.Dropout(dropout),
            nn.Linear(num_dense, num_dense),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_dense, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


soil_model = SoilCNN()
soil_model.load_state_dict(torch.load("soil_classif.pt", map_location='cuda'))
soil_model.eval()

soil_classes = ['alluvial', 'clay', 'loamy', 'black', 'red']

def predict_soil_type(image_path):
    transform = transforms.Compose([
        transforms.Resize((160, 160)), 
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = soil_model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        return soil_classes[predicted.item()]
    
    
def getData(N, P, K, temperature, humidity, ph, rainfall ):
    data = {
    'N': [N],
    'P': [P],
    'K': [K],
    'temperature': [temperature],
    'humidity': [humidity],
    'ph': [ph],
    'rainfall': [rainfall]
    }
    df=pd.DataFrame(data)
    df_scaled=mms.transform(df)
    df_scaled_df = pd.DataFrame(df_scaled, columns=df.columns)
    df_selected=selector.transform(df_scaled_df)
    y_pred=model.predict(df_selected)
    print("[INFO] y_pred: ",y_pred)
    predicted_crop = encoder.inverse_transform([y_pred[0]])[0]
    print("[INFO] Predicted crop:", predicted_crop)
    return predicted_crop


def soil_rules():
    
    soil_crop_map = {
    "alluvial": [
        "rice", "maize", "jute", "cotton", "banana", "lentil",
        "orange", "pomegranate", "papaya", "kidneybeans", "blackgram",
        "chickpea", "mungbean"
    ],
    "black": [
        "cotton", "rice", "maize", "pigeonpeas", "chickpea", "lentil",
        "grapes", "banana", "papaya", "blackgram", "mungbean"
    ],
    "red": [
        "cotton", "rice", "maize", "pigeonpeas", "mothbeans", "mungbean", 
        "chickpea", "banana", "papaya", "blackgram", "lentil"
    ],
    "clay": [
        "rice", "banana", "lentil", "blackgram", "maize"
    ],
    "loamy": [
        "apple", "orange", "grapes", "mango", "coconut",
        "pomegranate", "banana", "papaya", "watermelon", "muskmelon",
        "maize", "coffee", "lentil", "chickpea", "mungbean",
        "kidneybeans", "blackgram", "rice", "cotton"
    ]
}

    return soil_crop_map


# this gets the historic data from meteostat and return avg rain
def get_avg_rain(lat, lon):
    try:
        #print(f"lat:{lat} lon:{lon}")  # fro debugging
        station= nearby_stations(lat, lon)
        location = Point(station['latitude'], station['longitude'])
        start = datetime(2019, 1, 1)
        end = datetime.now()
   
        data = Monthly(location, start, end).fetch()
    except Exception as e:
        print("[ERROR] get_avg_rain func issue")
        return

    avg_rain = data['prcp'].mean()
    #print(f"in backend avg rain calc:{avg_rain}")  # fro debugging
    
    return avg_rain

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
    stations = Stations().nearby(lat, lon)
    station = stations.fetch(1)
    if station.empty:
        raise ValueError("No nearby station found")
    return station.iloc[0]