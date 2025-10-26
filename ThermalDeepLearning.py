import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms, models
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# ===============================
# CONFIG
# ===============================
DATA_DIR = r"C:\Users\testriad\Desktop\DeepLearning\dataset\datasets\thermal_classification_cropped"
SAVE_DIR = "results"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATIENCE = 5  # for early stopping

# ===============================
# DATA LOADING
# ===============================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Dataset path not found: {DATA_DIR}")

full_dataset = datasets.ImageFolder(DATA_DIR, transform=eval_transform)
print(f"Class distribution (full): {Counter(full_dataset.targets)}")

train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = eval_transform
test_dataset.dataset.transform = eval_transform

train_labels = [full_dataset.targets[i] for i in train_dataset.indices]
val_labels = [full_dataset.targets[i] for i in val_dataset.indices]
test_labels = [full_dataset.targets[i] for i in test_dataset.indices]
print(f"Train: {Counter(train_labels)} | Val: {Counter(val_labels)} | Test: {Counter(test_labels)}")

class_counts = np.bincount(train_labels)
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
samples_weights = [class_weights[label] for label in train_labels]
sampler = WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===============================
# MODEL SETUP (with Dropout)
# ===============================
from torchvision.models import resnet18, ResNet18_Weights
base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

num_features = base_model.fc.in_features
base_model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 1)
)
model = base_model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights[0] / class_weights[1]]).to(DEVICE))
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)  # L2 regularization
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# ===============================
# TRAINING & VALIDATION
# ===============================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.float().unsqueeze(1).to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = torch.sigmoid(outputs) > 0.5
        total_correct += (preds == labels.byte()).sum().item()
        total_samples += labels.size(0)
    return total_loss / len(loader), total_correct / total_samples

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.float().unsqueeze(1).to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.sigmoid(outputs)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_correct += ((preds > 0.5) == labels.byte()).sum().item()
            total_samples += labels.size(0)
    return total_loss / len(loader), total_correct / total_samples, np.array(all_labels), np.array(all_preds)

train_losses, val_losses, train_accs, val_accs = [], [], [], []
best_val_loss = float("inf")
early_stop_counter = 0

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion)
    scheduler.step(val_loss)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
          f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
    else:
        early_stop_counter += 1
        if early_stop_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

# ===============================
# TEST EVALUATION
# ===============================
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pth")))
test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion)
y_pred_labels = (y_pred > 0.5).astype(int)

print("\n=== TEST RESULTS ===")
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
print(classification_report(y_true, y_pred_labels, target_names=['non_icas', 'icas']))

# ===============================
# SAVE VISUALIZATIONS
# ===============================
# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_labels)
plt.figure(figsize=(5, 5))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"))
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.savefig(os.path.join(SAVE_DIR, "roc_curve.png"))
plt.close()

# Loss Curves
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title("Loss Curves")
plt.legend()
plt.savefig(os.path.join(SAVE_DIR, "loss_curve.png"))
plt.close()

# Accuracy Curves
plt.figure()
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.title("Accuracy Curves")
plt.legend()
plt.savefig(os.path.join(SAVE_DIR, "accuracy_curve.png"))
plt.close()

print(f"✅ All visualizations saved in: {os.path.abspath(SAVE_DIR)}")
