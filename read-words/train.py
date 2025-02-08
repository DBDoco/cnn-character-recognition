import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

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


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_epochs = 10
    batch_size = 32
    learning_rate = 0.001

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_csv = "train.csv"         
    train_images_dir = "train"        
    train_dataset = HandwrittenNameDataset(train_csv, train_images_dir, transform=transform)
    
    label2idx = train_dataset.label2idx
    num_classes = len(label2idx)
    print(f"Found {num_classes} classes.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] finished with average loss: {avg_loss:.4f}")

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'label2idx': label2idx
    }
    torch.save(checkpoint, "model.pth")
    print("Model saved as model.pth")


if __name__ == "__main__":
    train_model()