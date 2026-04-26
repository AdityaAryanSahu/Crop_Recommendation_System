
import pandas as pd
import numpy as np
import io
import joblib
from typing import Tuple, List
import torch
from torchvision import transforms
import torch.nn.functional as F
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
    
    
class ANN(nn.Module):
    def __init__(self, input_dim = 27, num_classes = 22, num_dense=64, dropout=0.2):
        super(ANN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, num_dense),
            nn.ReLU(),
            nn.BatchNorm1d(num_dense),
            nn.Dropout(dropout),
            nn.Linear(num_dense, num_dense),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_dense, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

class MultiModal:
    def __init__(self):
        try:
            current_file_path = os.path.dirname(os.path.abspath(__file__))
            MODEL_BASE_PATH = os.path.abspath(os.path.join(current_file_path, "..", "..", "ML_models"))
            print(f"[INFO] MODEL_BASE_PATH: {MODEL_BASE_PATH}")
            print(f"[INFO] Path exists: {os.path.exists(MODEL_BASE_PATH)}")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[INFO] Using device: {self.device}")
            
            # Check if directory exists
            if not os.path.exists(MODEL_BASE_PATH):
                raise FileNotFoundError(f"Model directory not found: {MODEL_BASE_PATH}")
            
            # List files in directory
            print(f"[INFO] Files in MODEL_BASE_PATH: {os.listdir(MODEL_BASE_PATH)}")
            
            # Load sklearn models
            print("[INFO] Loading voting_clf.pkl...")
            self.voting_clf_model = joblib.load(os.path.join(MODEL_BASE_PATH, "voting_clf.pkl"))
            
            print("[INFO] Loading all meta models...")
            self.xg_meta_model = joblib.load(os.path.join(MODEL_BASE_PATH, "xgboost.pkl"))
            self.rf_meta_model = joblib.load(os.path.join(MODEL_BASE_PATH, "random_forest.pkl"))
            self.lr_meta_model = joblib.load(os.path.join(MODEL_BASE_PATH, "logistic_regression.pkl"))
            
            self.ann_model = ANN().to(self.device)
            ann_path = os.path.join(MODEL_BASE_PATH, "best_ann_meta.pt")
            self.ann_model.load_state_dict(
                torch.load(ann_path, map_location=self.device, weights_only=True)
            )
            self.ann_model.eval()
            
            print("[INFO] all meta models loaded successfully")
            
            print("[INFO] Loading sc.pkl...")
            self.mms = joblib.load(os.path.join(MODEL_BASE_PATH, "sc.pkl"))
            
            print("[INFO] Loading selector.pkl...")
            self.selector = joblib.load(os.path.join(MODEL_BASE_PATH, "selector.pkl"))
            
            print("[INFO] Loading label_encoder.pkl...")
            self.encoder = joblib.load(os.path.join(MODEL_BASE_PATH, 'label_encoder.pkl'))
            
            # Setup PyTorch
            self.soil_classes = ['alluvial', 'clay', 'loamy', 'black', 'red']
           
            
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
           # self.soil_crop_map = self.soil_rules()
            
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
            predict_soil =  self.soil_classes[predicted.item()]
            soil_probs = F.softmax(outputs, dim=1).cpu().numpy()
            return predict_soil, soil_probs
        
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
        y_pred = self.voting_clf_model.predict(df_selected)
        y_pred_proba = self.voting_clf_model.predict_proba(df_selected)
        print("[INFO] y_pred (voting classif): ", y_pred)
        predicted_crop = self.encoder.inverse_transform([y_pred[0]])[0]
        print("[INFO] Predicted crop:", predicted_crop)
        return predicted_crop, y_pred_proba

    # def soil_rules(self):
    #     return {
    #         "alluvial": [
    #             "rice", "maize", "jute", "cotton", "banana", "lentil",
    #             "orange", "pomegranate", "papaya", "kidneybeans", "blackgram",
    #             "chickpea", "mungbean"
    #         ],
    #         "black": [
    #             "cotton", "rice", "maize", "pigeonpeas", "chickpea", "lentil",
    #             "grapes", "banana", "papaya", "blackgram", "mungbean"
    #         ],
    #         "red": [
    #             "cotton", "rice", "maize", "pigeonpeas", "mothbeans", "mungbean", 
    #             "chickpea", "banana", "papaya", "blackgram", "lentil"
    #         ],
    #         "clay": [
    #             "rice", "banana", "lentil", "blackgram", "maize"
    #         ],
    #         "loamy": [
    #             "apple", "orange", "grapes", "mango", "coconut",
    #             "pomegranate", "banana", "papaya", "watermelon", "muskmelon",
    #             "maize", "coffee", "lentil", "chickpea", "mungbean",
    #             "kidneybeans", "blackgram", "rice", "cotton"
    #         ]
    #     }

    def perform_fusion(self, crop_prob , soil_prob, predicted_soil ) -> Tuple[str, bool, List[str], str]:
        
        meta_input = np.hstack([soil_prob, crop_prob])
        meta_prob = self.xg_meta_model.predict_proba(meta_input)
        
        max_prob = np.max(meta_prob)
        if max_prob > 0.45:
            is_compatible = True
        else:
            is_compatible = False
        
        message = ""
        top_3_crops = []
        final_crop = ""
        
        if is_compatible:
            print(f"xg probs: {max_prob}")
            final_crop_idx = self.xg_meta_model.predict(meta_input)
            final_crop = self.encoder.inverse_transform([final_crop_idx])[0]
            message = f"Fusion Success: Raw crop '{final_crop.capitalize()}' is compatible with predicted {predicted_soil.capitalize()} soil."
        else:
            lr_probs = self.lr_meta_model.predict_proba(meta_input)
            rf_probs = self.rf_meta_model.predict_proba(meta_input)
            meta_tensor = torch.tensor(meta_input, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                outputs = self.ann_model(meta_tensor)
                ann_probs = F.softmax(outputs, dim=1).cpu().numpy()
            
            weights = np.array([0.4, 0.3, 0.2, 0.1]) #XGB, ANN, RF, LR
            weighted_probs = (meta_prob * weights[0] + 
                  ann_probs * weights[1] + 
                  rf_probs * weights[2] + 
                  lr_probs * weights[3])
            top_3_indices = np.argsort(weighted_probs[0])[-3:][::-1]
            top_3_crops = self.encoder.inverse_transform(top_3_indices)
            
            message = f"Fusion Fallback: Raw prediction was incompatible. Recommended top compatible crop: {top_3_crops[0]}."

        compatible_alternatives = [c.capitalize() for c in top_3_crops[:3]]
        return final_crop.capitalize(), is_compatible, compatible_alternatives, message
