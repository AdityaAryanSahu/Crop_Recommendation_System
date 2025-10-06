
import pandas as pd
import numpy as np
import io
import joblib
from typing import Tuple, List
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import os
import traceback

# Use relative path or environment variable
MODEL_BASE_PATH = "/app/my_app/ML_models"

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

class MultiModal:
    def __init__(self):
        try:
            print(f"[INFO] MODEL_BASE_PATH: {MODEL_BASE_PATH}")
            print(f"[INFO] Path exists: {os.path.exists(MODEL_BASE_PATH)}")
            
            # Check if directory exists
            if not os.path.exists(MODEL_BASE_PATH):
                raise FileNotFoundError(f"Model directory not found: {MODEL_BASE_PATH}")
            
            # List files in directory
            print(f"[INFO] Files in MODEL_BASE_PATH: {os.listdir(MODEL_BASE_PATH)}")
            
            # Load sklearn models
            print("[INFO] Loading voting_clf.pkl...")
            self.model = joblib.load(os.path.join(MODEL_BASE_PATH, "voting_clf.pkl"))
            
            print("[INFO] Loading sc.pkl...")
            self.mms = joblib.load(os.path.join(MODEL_BASE_PATH, "sc.pkl"))
            
            print("[INFO] Loading selector.pkl...")
            self.selector = joblib.load(os.path.join(MODEL_BASE_PATH, "selector.pkl"))
            
            print("[INFO] Loading label_encoder.pkl...")
            self.encoder = joblib.load(os.path.join(MODEL_BASE_PATH, 'label_encoder.pkl'))
            
            # Setup PyTorch
            self.soil_classes = ['alluvial', 'clay', 'loamy', 'black', 'red']
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[INFO] Using device: {self.device}")
            
            # Load CNN model
            print("[INFO] Initializing SoilCNN...")
            self.soil_model = SoilCNN()
            
            cnn_path = os.path.join(MODEL_BASE_PATH, "soil_classif.pt")
            print(f"[INFO] Loading CNN weights from: {cnn_path}")
            print(f"[INFO] CNN file exists: {os.path.exists(cnn_path)}")
            
            # Load with weights_only=True for security (PyTorch 2.0+)
            try:
                self.soil_model.load_state_dict(
                    torch.load(cnn_path, map_location=self.device, weights_only=True)
                )
            except TypeError:
                # Fallback for older PyTorch versions
                self.soil_model.load_state_dict(
                    torch.load(cnn_path, map_location=self.device)
                )
            
            self.soil_model.eval()
            self.soil_model.to(self.device)
            print("[INFO] CNN model loaded successfully")
            
            # Setup transforms
            self.transform = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
            ])
            
            # Load soil rules
            self.soil_crop_map = self.soil_rules()
            
            print("[SUCCESS] All models loaded successfully!")
            
        except FileNotFoundError as e:
            print(f"[ERROR] File not found: {e}")
            traceback.print_exc()
            raise RuntimeError(f"[ERROR] Model file not found: {e}")
        except Exception as e:
            print(f"[ERROR] Model loading failed: {e}")
            print(f"[ERROR] Error type: {type(e).__name__}")
            traceback.print_exc()
            raise RuntimeError(f"[ERROR] Model loading error: {e}")

    def predict_soil_type(self, image_file_content: bytes):
        image = Image.open(io.BytesIO(image_file_content)).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.soil_model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            return self.soil_classes[predicted.item()]
        
    def crop_pred(self, N, P, K, temperature, Humidity, ph, rainfall):
        data = {
            'N': [N],
            'P': [P],
            'K': [K],
            'temperature': [temperature],
            'humidity': [Humidity],
            'ph': [ph],
            'rainfall': [rainfall]
        }
        df = pd.DataFrame(data)
        df_scaled = self.mms.transform(df)
        df_scaled_df = pd.DataFrame(df_scaled, columns=df.columns)
        df_selected = self.selector.transform(df_scaled_df)
        y_pred = self.model.predict(df_selected)
        print("[INFO] y_pred: ", y_pred)
        predicted_crop = self.encoder.inverse_transform([y_pred[0]])[0]
        print("[INFO] Predicted crop:", predicted_crop)
        return predicted_crop

    def soil_rules(self):
        return {
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

    def perform_fusion(self, raw_crop: str, predicted_soil: str) -> Tuple[str, bool, List[str], str]:
        valid_crops = self.soil_crop_map.get(predicted_soil, [])
        is_compatible = raw_crop.lower() in [c.lower() for c in valid_crops]
        
        final_crop = raw_crop
        message = ""
        
        if is_compatible:
            message = f"Fusion Success: Raw crop '{raw_crop.capitalize()}' is compatible with predicted {predicted_soil.capitalize()} soil."
        else:
            final_crop = valid_crops[0].capitalize() if valid_crops else "None (No compatible crops found)"
            message = f"Fusion Fallback: Raw prediction was incompatible. Recommended top compatible crop: {final_crop}."

        compatible_alternatives = [c.capitalize() for c in valid_crops[:5]]
        return final_crop.capitalize(), is_compatible, compatible_alternatives, message
