# =========================
# IMPORTS
# =========================
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# =========================
# LOAD MODELS
# =========================
vc = joblib.load(r"C:\Users\Lenovo\Downloads\voting_clf.pkl")
selector = joblib.load(r"D:\Crop_rec_system\my_app\ML_models\selector.pkl")
encoder = joblib.load(r"D:\Crop_rec_system\my_app\ML_models\label_encoder.pkl")
sc = joblib.load(r"D:\Crop_rec_system\my_app\ML_models\sc.pkl")

# FIXED PATHS ✅
xgb_meta_clf = joblib.load(r"D:\Crop_rec_system\xgboost.pkl")
lr_meta_clf = joblib.load(r"D:\Crop_rec_system\logistic_regression.pkl")
rf_meta_clf = joblib.load(r"D:\Crop_rec_system\random_forest.pkl")
ann_meta_clf = joblib.load(r"D:\Crop_rec_system\ann.pkl")

# =========================
# LOAD TEST DATASET
# =========================
df = pd.read_csv(r"D:\Crop_rec_system\my_app\ML_modules\meta_test_dataset.csv")

X = df.drop('label', axis=1)
y = df['label']

# Split features
soil_cols = ['soil_prob_alluvial','soil_prob_black','soil_prob_clay','soil_prob_loamy','soil_prob_red']
base_cols = ['N','P','K','temperature','humidity','ph','rainfall']

X_test_base = X[base_cols]
soil_probs_test = X[soil_cols]

# Encode labels
y_test = encoder.transform(y)

# =========================
# BASE MODEL PROCESSING
# =========================
X_test_scaled = sc.transform(X_test_base)
X_test_selected = selector.transform(X_test_scaled)

vc_test_probs = vc.predict_proba(X_test_selected)

# Meta features
meta_X_test = np.hstack([soil_probs_test.values, vc_test_probs])

# =========================
# EVALUATE META MODELS
# =========================
num_classes = len(encoder.classes_)
input_dim = meta_X_test.shape[1]

class ANN(nn.Module):
    def __init__(self, input_dim, num_classes, num_dense=64, dropout=0.2):
        super(ANN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, num_dense), # Corrected input_dim
            nn.ReLU(),
            nn.BatchNorm1d(num_dense),
            nn.Dropout(dropout),
            nn.Linear(num_dense, num_dense),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_dense, num_classes) # Corrected output_dim
        )

    def forward(self, x):
        return self.classifier(x)
    
print(f"{input_dim} && {num_classes}")    
ann_model = ANN(input_dim, num_classes).to('cuda')
ann_model.load_state_dict(
    torch.load(r"D:\Crop_rec_system\best_ann_meta.pt", map_location='cuda')
)
ann_model.eval()
models = {
    "XGBoost": xgb_meta_clf,
    "Logistic Regression": lr_meta_clf,
    "Random Forest": rf_meta_clf,
    "ANN": ann_model
}

print("\n===== META MODEL PERFORMANCE =====\n")

for name, model in models.items():
    if name == "ANN":
        continue
    preds = model.predict(meta_X_test)

    print(f"\n{name}")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"F1 Score: {f1_score(y_test, preds, average='weighted'):.4f}")
    print(classification_report(y_test, preds))

# =========================
# CNN MODEL (SOIL)
# =========================
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
        return self.classifier(x)
    
    
    


# =========================
# SINGLE INPUT + IMAGE TEST
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

soil_model = SoilCNN().to(device)
soil_model.load_state_dict(
    torch.load(r"D:\Crop_rec_system\my_app\ML_models\soil_classif.pt", map_location=device)
)
soil_model.eval()

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])

# ===== USER INPUT =====
data = {
'N': [100], 'P': [20], 'K': [30],
    'temperature': [22], 'humidity': [88], 'ph': [5.8], # Acidic pH
    'rainfall': [180]
}

image_path = r"D:\Soil_type_dataset\Loamy\loamy 4.png"

# Tabular processing
df_input = pd.DataFrame(data)
df_scaled = sc.transform(df_input)
df_selected = selector.transform(df_scaled)
vc_pred = vc.predict(df_selected) 
crop = encoder.inverse_transform([vc_pred[0]])[0]
vc_probs = vc.predict_proba(df_selected)
max_prob = np.max(vc_probs)
pred_idx = np.argmax(vc_probs)
crop = encoder.inverse_transform([pred_idx])[0]
print(f"raw pred by voting classifier: {crop} ({max_prob*100:.2f}% confidence)") 

vc_probs = vc.predict_proba(df_selected)

# Image processing
image = Image.open(image_path).convert('RGB')
image_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = soil_model(image_tensor)
    soil_probs = F.softmax(outputs, dim=1).cpu().numpy()

# Meta input
meta_input = np.hstack([soil_probs, vc_probs])




print("\n===== FINAL PREDICTIONS =====\n")

meta_input_tensor = torch.tensor(meta_input, dtype=torch.float32).to(device)

for name, model in models.items():
    if name == "ANN":
        #pyTorch Inference Logic
        model.eval()
        with torch.no_grad():
            outputs = model(meta_input_tensor)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
    else:
        #sklearn/XGBoost Inference Logic
        probs = model.predict_proba(meta_input)

    max_prob = np.max(probs)
    pred_idx = np.argmax(probs)
    crop = encoder.inverse_transform([pred_idx])[0]
    print(f"{name}: {crop} ({max_prob*100:.2f}% confidence)")
