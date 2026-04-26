import os
import cv2
import numpy as np
import pickle, gzip
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random

print("started...")
base_path = r"D:\Soil_type_dataset"
classes = ['alluvial', 'clay', 'loamy', 'black', 'red']
img_size = (160, 160)

X, y = [], []

def augment_image(img):
    # Random horizontal flip
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
    # Random rotation
    angle = random.choice([0, 90, 180, 270])
    img = cv2.rotate(img, {
        0: cv2.ROTATE_180,
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE
    }[angle])
    return img

for idx, label in enumerate(classes):
    folder = os.path.join(base_path, label)
    count = 0
    aug_count=0
    for file in tqdm(os.listdir(folder), desc=f"Loading {label}"):
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            X.append(img)
            y.append(idx)
            count += 1
            if label == 'alluvial':
                continue
            # Data augmentation (1x per image for now)
            aug_img = augment_image(img)
            X.append(aug_img)
            y.append(idx)
            aug_count += 1
    print(f"{label} done: {count} original + {aug_count} augmented = {count*2} total")

print("Preprocessing done.")
X = np.array(X, dtype=np.uint8)  
X = X.astype(np.float32) / 255.0  
y = np.array(y)

print("split started.")
# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Save
with gzip.open("soil_data.pkl.gz", "wb") as f:
    pickle.dump((X_train, X_val, y_train, y_val), f)

print(f"Final shape: Train: {X_train.shape}, Val: {X_val.shape}")
print("Data saved successfully.")
