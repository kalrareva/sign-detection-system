"""
Trains a 1D Convolutional Neural Network (CNN) on hand landmark features.

Usage:
    python src/train_cnn.py
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    LANDMARKS_CSV_PATH,
    CNN_MODEL_PATH,
    CNN_CLASSES_PATH,
    CNN_CONFUSION_MATRIX_PATH,
    CNN_TRAINING_CURVES_PATH,
    MODEL_DIR,
)
from model_cnn import SignLandmarkCNN


class LandmarkDataset(Dataset):
    """Dataset for hand landmarks with optional geometric augmentation."""

    def __init__(self, features, labels, augment=False):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx].clone()  # Shape: (63,)
        y = self.labels[idx]

        if self.augment:
            # Reshape to (21, 3) for geometric manipulations
            coords = x.view(21, 3)

            # 1. Random Scaling (0.90 to 1.10)
            scale = np.random.uniform(0.90, 1.10)
            coords = coords * scale

            # 2. Random 2D in-plane rotation (-15 to +15 deg)
            angle = np.radians(np.random.uniform(-15, 15))
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot_mat = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a,  cos_a, 0],
                [0,      0,     1]
            ], dtype=torch.float32)
            coords = torch.matmul(coords, rot_mat.T)

            # 3. Small Gaussian jitter
            noise = torch.randn_like(coords) * 0.015
            coords = coords + noise

            x = coords.view(-1)

        # Reshape to (3, 21): 3 channels (x, y, z) across 21 landmarks
        x_cnn = x.view(21, 3).transpose(0, 1)
        return x_cnn, y


def train_cnn():
    if not os.path.exists(LANDMARKS_CSV_PATH):
        print(f"Dataset not found at {LANDMARKS_CSV_PATH}. Run collect_data.py first.")
        sys.exit(1)

    df = pd.read_csv(LANDMARKS_CSV_PATH)
    if df.empty:
        print("Dataset is empty. Run collect_data.py first.")
        sys.exit(1)

    labels_sorted = sorted(df["label"].unique())
    label_to_idx = {l: i for i, l in enumerate(labels_sorted)}
    num_classes = len(labels_sorted)

    print(f"\nLoaded {len(df)} samples across {num_classes} classes: {labels_sorted}")
    print(df["label"].value_counts().sort_index())

    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = np.array([label_to_idx[l] for l in df["label"]])

    # Train / Val / Test split (70% train, 15% val, 15% test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    train_ds = LandmarkDataset(X_train, y_train, augment=True)
    val_ds = LandmarkDataset(X_val, y_val, augment=False)
    test_ds = LandmarkDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training CNN model on device: {device}")

    model = SignLandmarkCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

    num_epochs = 40
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_model_weights = None

    print(f"\n--- Training CNN for {num_epochs} Epochs ---")
    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()
        running_train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)

        scheduler.step()
        train_loss = running_train_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        val_loss = running_val_loss / total
        val_acc = correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == num_epochs:
            print(f"Epoch [{epoch:2d}/{num_epochs:2d}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    # Load best weights
    model.load_state_dict(best_model_weights)
    model.to(device)

    # Final Evaluation on Held-out Test Set
    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_pred_all.extend(preds)
            y_true_all.extend(targets.numpy())

    print("\n" + "=" * 50)
    print("=== CNN Classification Report (Held-Out Test Set) ===")
    print("=" * 50)
    print(classification_report(y_true_all, y_pred_all, target_names=labels_sorted))

    # Save outputs
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Save CNN model weights
    torch.save(best_model_weights, CNN_MODEL_PATH)
    print(f"Saved trained CNN model weights to: {CNN_MODEL_PATH}")

    # 2. Save class names
    with open(CNN_CLASSES_PATH, "w") as f:
        json.dump(labels_sorted, f, indent=2)
    print(f"Saved class labels to: {CNN_CLASSES_PATH}")

    # 3. Save Confusion Matrix plot
    cm = confusion_matrix(y_true_all, y_pred_all)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_sorted)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
    plt.title("CNN Sign Language Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(CNN_CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()
    print(f"Saved CNN confusion matrix plot to: {CNN_CONFUSION_MATRIX_PATH}")

    # 4. Save Training Curves plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = range(1, num_epochs + 1)
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="royalblue", lw=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="coral", lw=2)
    ax1.set_title("Training & Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(epochs, [acc * 100 for acc in history["val_acc"]], label="Val Accuracy", color="forestgreen", lw=2)
    ax2.set_title("Validation Accuracy (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(CNN_TRAINING_CURVES_PATH, dpi=150)
    plt.close()
    print(f"Saved CNN training curves to: {CNN_TRAINING_CURVES_PATH}")


if __name__ == "__main__":
    train_cnn()
