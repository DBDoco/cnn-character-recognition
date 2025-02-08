import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class HandwrittenNameDataset(Dataset):
    def __init__(self, csv_file, images_dir, transform=None, label2idx=None):
        self.data = pd.read_csv(csv_file)
        self.data = self.data.dropna(subset=['IDENTITY'])
        self.images_dir = images_dir
        self.transform = transform

        if label2idx is None:
            labels = sorted(self.data['IDENTITY'].unique())
            self.label2idx = {label: idx for idx, label in enumerate(labels)}
        else:
            self.label2idx = label2idx
            self.data = self.data[self.data['IDENTITY'].isin(self.label2idx.keys())]

        self.data['label_idx'] = self.data['IDENTITY'].map(self.label2idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row['FILENAME']
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = row['label_idx']
        return image, label

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load("model.pth", map_location=device)
    label2idx = checkpoint['label2idx']
    num_classes = len(label2idx)
    print(f"Evaluating a model with {num_classes} classes.")

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    valid_csv = "valid.csv"      
    valid_images_dir = "valid"      
    valid_dataset = HandwrittenNameDataset(valid_csv, valid_images_dir, transform=transform, label2idx=label2idx)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print("\n📌 Model Performance:")
    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall: {recall:.4f}")
    print(f"✅ F1-score: {f1:.4f}")

if __name__ == "__main__":
    evaluate_model()