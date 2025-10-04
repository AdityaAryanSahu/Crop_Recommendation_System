
import random
from user_interface_backend import getData

def generate_simulated_input():
    return {
        'N': random.randint(0, 100),
        'P': random.randint(0, 100),
        'K': random.randint(0, 100),
        'temperature': round(random.uniform(10, 45), 2),
        'humidity': round(random.uniform(30, 100), 2),
        'ph': round(random.uniform(4.5, 8.5), 2),
        'rainfall': round(random.uniform(0, 250), 2)
    }

def get_simulated_prediction():
    data = generate_simulated_input()
    prediction = getData(
        data['N'], data['P'], data['K'],
        data['temperature'], data['humidity'],
        data['ph'], data['rainfall']
    )
    return data, prediction[0]
