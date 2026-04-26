import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report
import pickle, gzip
import numpy as np
import matplotlib.pyplot as plt
import time
import csv
from torch.amp import autocast, GradScaler

# cuDNN autotuner
torch.backends.cudnn.benchmark = True

# Load preprocessed dataset
with gzip.open("soil_data.pkl.gz", "rb") as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if device.type == "cuda":
    print("Using GPU:", torch.cuda.get_device_name(0))

# Preload data to GPU
X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

def preprocess_data(X, y, batch_size):
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=False)
    return loader

class SoilCNN(nn.Module):
    def __init__(self, num_classes=5, num_dense=128, dropout=0.5):
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

def train_model(X_train, y_train, X_test, y_test, epoch, batch_size, lr, num, drop):
    model = SoilCNN(num_classes=5, num_dense=num, dropout=drop).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()

    train_loader = preprocess_data(X_train, y_train, batch_size)
    test_loader = preprocess_data(X_test, y_test, batch_size)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for ep in range(epoch):
        start_epoch = time.time()
        model.train()
        train_loss, correct, total = 0, 0, 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            with autocast('cuda'):
                out = model(xb)
                loss = criterion(out, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * xb.size(0)
            _, preds = torch.max(out, 1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        train_acc = correct / total
        train_loss /= total

        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                with autocast(device_type='cuda'):
                    out = model(xb)
                    loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
                _, preds = torch.max(out, 1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total
        val_loss /= total
        print(f"Epoch {ep+1}/{epoch} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Time: {round(time.time() - start_epoch, 2)}s")
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

    return model, history

# === HYPERPARAMETER SEARCH ===
if __name__ == "__main__":
    print("Started tuning...")

    epochs = 15  # shorter for testing
    batch_size = 32

    lr_list = [0.001, 0.0008, 0.0005]
    drop_list = [0.2, 0.3, 0.4, 0.5]
    num_dense_list = [64, 128, 256]

    best_f1 = 0
    best_config = None
    best_model = None
    best_history = None

    os.makedirs("models", exist_ok=True)

    # CSV log file
    with open("soil_hyperparam_results.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["lr", "dropout", "dense", "f1_score", "train_time"])

    for lr in lr_list:
        for drop in drop_list:
            for num in num_dense_list:
                print(f"\n Training with lr={lr}, dropout={drop}, dense={num}")
                torch.cuda.empty_cache()
                start = time.time()
                model, history = train_model(X_train, y_train, X_test, y_test, epochs, batch_size, lr, num, drop)
                train_time = time.time() - start

                # Evaluation
                model.eval()
                all_preds = []
                with torch.no_grad():
                    for xb, _ in DataLoader(TensorDataset(X_test, y_test), batch_size=128):
                        out = model(xb.to(device))
                        preds = torch.argmax(out, dim=1)
                        all_preds.extend(preds.cpu().numpy())
                all_preds = np.array(all_preds)
                true_labels = y_test.cpu().numpy()

                f1 = round(f1_score(true_labels, all_preds, average='weighted'), 3)
                print(f" Weighted F1 Score: {f1} | Time: {round(train_time, 2)}s")

                # Log result
                with open("soil_hyperparam_results.csv", "a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([lr, drop, num, f1, round(train_time, 2)])

                if f1 > best_f1:
                    best_f1 = f1
                    best_config = (lr, drop, num)
                    
                # Free memory
                del model
                torch.cuda.empty_cache()

    print("\n Grid Search Completed!")
    print(f" Best Hyperparameters: lr={best_config[0]}, dropout={best_config[1]}, dense={best_config[2]}")
    print(f"Best Weighted F1 Score: {best_f1}")
