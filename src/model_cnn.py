"""
1D Convolutional Neural Network (CNN) architecture for Hand Landmark Classification.

Landmarks are structured as (B, Channels=3, Length=21), where Channels are (x, y, z)
and Length is the 21 MediaPipe hand landmarks in topological order along the kinematic chain.
1D convolutions extract spatial curvature and inter-joint geometric relations.
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn


class SignLandmarkCNN(nn.Module):
    """1D-CNN classifier for 21 3D hand landmarks."""

    def __init__(self, num_classes=12, in_channels=3, num_landmarks=21, dropout_rate=0.3):
        super().__init__()
        self.num_classes = num_classes

        # Block 1: Feature extraction along landmark sequence
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2)  # 21 -> 10
        )

        # Block 2: Higher-level geometric pattern extraction
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)  # (B, 128, 1)
        )

        # Fully connected classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Expects input shape: (B, 3, 21) or (B, 63)
        if x.dim() == 2 and x.shape[1] == 63:
            # Reshape from flat 63 (x0,y0,z0, x1,y1,z1, ...) to (B, 3, 21)
            x = x.view(-1, 21, 3).transpose(1, 2)
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out = self.classifier(out)
        return out


class CNNModelWrapper:
    """Convenience inference wrapper providing scikit-learn-like predict_proba interface."""

    def __init__(self, model_path, classes_path, device="cpu"):
        self.device = torch.device(device)
        with open(classes_path, "r") as f:
            self.classes_ = json.load(f)

        self.model = SignLandmarkCNN(num_classes=len(self.classes_))
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    def predict_proba(self, features):
        """Features: numpy array of shape (N, 63) or (63,).

        Returns:
            numpy array of shape (N, num_classes) containing probabilities.
        """
        if isinstance(features, list):
            features = np.array(features, dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        x = torch.tensor(features, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs
