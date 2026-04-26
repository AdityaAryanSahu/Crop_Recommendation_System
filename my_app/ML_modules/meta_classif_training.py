# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import MinMaxScaler
# import joblib


# df = pd.read_csv("updated_synthetic_meta_dataset.csv")

# from sklearn.model_selection import train_test_split
# X=df.drop('label', axis=1)
# y=df['label']
# X_train, x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
# X_train_base= X_train.drop(['soil_prob_alluvial','soil_prob_black','soil_prob_clay','soil_prob_loamy','soil_prob_red'], axis=1)
# X_test_base= x_test.drop(['soil_prob_alluvial','soil_prob_black','soil_prob_clay','soil_prob_loamy','soil_prob_red'], axis=1)
# soil_probs_train= X_train.drop(['N','P','K','temperature', 'humidity','ph','rainfall'], axis=1)
# soil_probs_test= x_test.drop(['N','P','K','temperature', 'humidity','ph','rainfall'], axis=1)

# from sklearn.preprocessing import LabelEncoder

# le = LabelEncoder()
# y_train_enc = le.fit_transform(y_train)
# y_test_enc = le.transform(y_test)

# clf1 = LogisticRegression(max_iter=1000)
# clf2 = KNeighborsClassifier()
# clf3 = DecisionTreeClassifier()
# clf4 = RandomForestClassifier(n_estimators=100)
# clf5 = SVC(probability=True)
# clf6 = GaussianNB()
# clf7 = XGBClassifier( eval_metric='mlogloss')


# vc = VotingClassifier(
#     estimators=[
#         ('lr', clf1),
#         ('knn', clf2),
#         ('dt', clf3),
#         ('rf', clf4),
#         ('svm', clf5),
#         ('nb', clf6),
#         ('xgb', clf7)
#     ],
#     voting='soft')

# skf = StratifiedKFold(n_splits=4)

# for train_idx, val_idx in skf.split(X_train_base,y_train_enc):
#   X_train, X_val = X_train_base.iloc[train_idx], X_train_base.iloc[val_idx]
#   y_train_new, y_val = y_train_enc[train_idx], y_train_enc[val_idx]

#   sc=MinMaxScaler()
#   x_train_scaled=sc.fit_transform(X_train)
#   x_val_scaled=sc.transform(X_val)
#   print(x_train_scaled.shape)
#   selector=SelectKBest(mutual_info_classif,k=7)
#   selector.fit_transform(x_train_scaled,y_train_new)
#   x_train_new=selector.transform(x_train_scaled)
#   x_val_new=selector.transform(x_val_scaled)


#   vc.fit(x_train_new, y_train_new)

#   vc_oof_probs[val_idx] = vc.predict_proba(x_val_new)
  
  
# meta_X_train = np.hstack([soil_probs_train, vc_oof_probs])
# meta_y_train = y_train_enc

# xgb_meta_clf = xgb.XGBClassifier(
#     max_depth=4,
#     n_estimators=200,
#     learning_rate=0.05,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     objective="multi:softprob",
#     num_class=num_crops
# )

# xgb_meta_clf.fit(meta_X_train, meta_y_train)


# =========================
# IMPORTS
# =========================
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report

from torch.amp import autocast, GradScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import time
from sklearn.metrics import f1_score

from xgboost import XGBClassifier

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(r"D:\Crop_rec_system\my_app\ML_modules\meta_test_dataset.csv")

X = df.drop('label', axis=1)
y = df['label']

# Train-test split
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Split base + soil probs
soil_cols = ['soil_prob_alluvial','soil_prob_black','soil_prob_clay','soil_prob_loamy','soil_prob_red']
base_cols = ['N','P','K','temperature','humidity','ph','rainfall']

X_train_base = X_train_full[base_cols]
X_test_base = X_test_full[base_cols]

soil_probs_train = X_train_full[soil_cols]
soil_probs_test = X_test_full[soil_cols]

# Encode labels
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

num_classes = len(le.classes_)

# =========================
# BASE MODELS (Voting)
# =========================
clf1 = LogisticRegression(max_iter=1000)
clf2 = KNeighborsClassifier()
clf3 = DecisionTreeClassifier()
clf4 = RandomForestClassifier(n_estimators=100, random_state=42)
clf5 = SVC(probability=True)
clf6 = GaussianNB()
clf7 = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)

vc = VotingClassifier(
    estimators=[
        ('lr', clf1),
        ('knn', clf2),
        ('dt', clf3),
        ('rf', clf4),
        ('svm', clf5),
        ('nb', clf6),
        ('xgb', clf7)
    ],
    voting='soft'
)

# =========================
# OUT-OF-FOLD PREDICTIONS
# =========================
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

vc_oof_probs = np.zeros((X_train_base.shape[0], num_classes))
val_scores = []

for train_idx, val_idx in skf.split(X_train_base, y_train_enc):
    X_tr = X_train_base.iloc[train_idx]
    X_val = X_train_base.iloc[val_idx]
    y_tr = y_train_enc[train_idx]
    y_val = y_train_enc[val_idx]

    # Scaling
    scaler = MinMaxScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)

    # Feature Selection
    selector = SelectKBest(mutual_info_classif, k=7)
    X_tr_sel = selector.fit_transform(X_tr_scaled, y_tr)
    X_val_sel = selector.transform(X_val_scaled)

    # Train base model
    vc.fit(X_tr_sel, y_tr)

    # Store OOF probs
    vc_oof_probs[val_idx] = vc.predict_proba(X_val_sel)

    # Validation score
    val_pred = vc.predict(X_val_sel)
    val_scores.append(accuracy_score(y_val, val_pred))

print("\nBase Model CV Accuracy:", np.mean(val_scores))

# =========================
# FINAL BASE MODEL TRAINING
# =========================
scaler_final = MinMaxScaler()
X_train_scaled = scaler_final.fit_transform(X_train_base)

selector_final = SelectKBest(mutual_info_classif, k=7)
X_train_sel = selector_final.fit_transform(X_train_scaled, y_train_enc)

vc.fit(X_train_sel, y_train_enc)

# =========================
# CREATE META TRAIN DATA
# =========================
meta_X_train = np.hstack([soil_probs_train.values, vc_oof_probs])
meta_y_train = y_train_enc

X_test_scaled = scaler_final.transform(X_test_base)
X_test_sel = selector_final.transform(X_test_scaled)
vc_test_probs = vc.predict_proba(X_test_sel)

meta_X_test = np.hstack([soil_probs_test.values, vc_test_probs])

# =========================
# META MODELS
# =========================

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

# =========================
# IMPROVED PREPROCESSING
# =========================
def prepare_loaders(X, y, batch_size):
    # Convert to Tensors and ensure correct types
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=False)  # GPU data: no CPU pinning needed
    return loader

# =========================
# CORRECTED TRAINING CALL
# =========================
input_dim = meta_X_train.shape[1] # This should be 27
batch_size = 64

train_loader = prepare_loaders(meta_X_train, meta_y_train, batch_size)
test_loader = prepare_loaders(meta_X_test, y_test_enc, batch_size)

def train_model(train_loader, test_loader, epoch, lr, input_dim, num_classes, num_dense, drop):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ANN(input_dim, num_classes, num_dense, drop).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler() # Matches your 'cuda' autocast logic

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for ep in range(epoch):
        start_epoch = time.time()
        model.train()
        train_loss, correct, total = 0, 0, 0

        # training 
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
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
                xb = xb.to(device)
                yb = yb.to(device)
                with autocast(device_type='cuda'):
                    out = model(xb)
                    loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
                _, preds = torch.max(out, 1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        val_acc = correct / total
        val_loss /= total

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {ep+1}/{epoch} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Time: {round(time.time() - start_epoch, 2)}s")

    return model, history

# Execute
final_ann, history = train_model(train_loader, test_loader, 10, 0.0005, input_dim, num_classes, 64, 0.5)


meta_models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),

    "Logistic Regression": LogisticRegression(max_iter=1000),

    "XGBoost": XGBClassifier(
        max_depth=4,
        n_estimators=200,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric='mlogloss'
    ),
}



final_ann.eval()
all_preds = []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to('cuda')
        out = final_ann(xb)
        preds = torch.argmax(out, dim=1)
        all_preds.extend(preds.cpu().numpy())

    # Convert predictions to array
    all_preds = np.array(all_preds)
# Train meta models
for name, model in meta_models.items():
    model.fit(meta_X_train, meta_y_train)

# =========================
# PREPARE TEST META FEATURES
# =========================
lr_list = [0.001, 0.0008, 0.0005]
drop_list = [0.2, 0.3, 0.4, 0.5]
num_dense_list = [64, 128, 256]

best_f1 = -1.0
best_params = {}
best_model_state = None

for lr in lr_list:
        for drop in drop_list:
            for num in num_dense_list:
                print(f"\n Training with lr={lr}, dropout={drop}, dense={num}")
                torch.cuda.empty_cache()
                start = time.time()
                model, history = train_model(train_loader, test_loader, 10, lr, input_dim, num_classes, num, drop)
                train_time = time.time() - start

                # Evaluation
                model.eval()
                all_preds = []
                true_labels= []
                with torch.no_grad():
                    for xb, yb in test_loader:
                        out = model(xb.to('cuda'))
                        preds = torch.argmax(out, dim=1)
                        all_preds.extend(preds.cpu().numpy())
                        true_labels.extend(yb.numpy())
                all_preds = np.array(all_preds)
                #true_labels = y_test.cpu().numpy()

                f1 = round(f1_score(true_labels, all_preds, average='weighted'), 3)
                print(f" Weighted F1 Score: {f1} | Time: {round(train_time, 2)}s")
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = {'lr': lr, 'dropout': drop, 'dense': num}
                    # Deep copy the model state so it's not deleted by 'del model'
                    import copy
                    best_model_state = copy.deepcopy(model.state_dict())
                    print(f" New Best Model Found! F1: {best_f1}")

                # Free memory
                del model
                torch.cuda.empty_cache()
                
                
print("\n" + "="*30)
print("GRID SEARCH COMPLETE")
print(f"Best F1 Score: {best_f1}")
print(f"Best Parameters: {best_params}")
print("="*30)

# Save the best model found during the search
if best_model_state:
    torch.save(best_model_state, r"D:\Crop_rec_system\best_ann_meta.pt")
    print("Best model weights saved successfully.")


# =========================
# EVALUATION
# =========================
#print("\n===== META MODEL PERFORMANCE =====\n")

#for name, model in meta_models.items():
    #preds = model.predict(meta_X_test)
    #acc = accuracy_score(y_test_enc, preds)

    #print(f"{name} Accuracy: {acc:.4f}")
    #print(classification_report(y_test_enc, preds))
    #print("-" * 50)
    #filename = name.lower().replace(" ", "_") + ".pkl"
    #joblib.dump(model, filename)
    
    
