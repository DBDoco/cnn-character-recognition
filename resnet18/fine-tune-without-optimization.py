import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

def load_emnist_mapping():
    import numpy as np
    mapping_path = "../emnist-balanced-mapping.txt"
    mapping = np.loadtxt(mapping_path, delimiter=" ", dtype=int)
    return {i: chr(mapping[i][1]) for i in range(len(mapping))}

label_mapping = load_emnist_mapping()
print("✅ Loaded EMNIST Label Mapping")

pretrained_model = models.resnet18(pretrained=True)

pretrained_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

num_ftrs = pretrained_model.fc.in_features
pretrained_model.fc = nn.Linear(num_ftrs, 47)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_model = pretrained_model.to(device)

transform_train = transforms.Compose([
    transforms.Grayscale(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(10, translate=(0.1, 0.1), shear=5, scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_test = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.EMNIST(
    root="./data", split="balanced", train=True, download=True, transform=transform_train
)
test_dataset = torchvision.datasets.EMNIST(
    root="./data", split="balanced", train=False, download=True, transform=transform_test
)

batch_size = 60
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pretrained_model.parameters(), lr=0.0001) 

num_epochs = 100
best_loss = float("inf")
early_stop_patience = 3
epochs_no_improve = 0

print("\n🚀 Starting Training...\n")

for epoch in range(1, num_epochs + 1):
    pretrained_model.train()
    running_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = pretrained_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss / len(train_loader))

    train_loss = running_loss / len(train_loader)

    pretrained_model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = pretrained_model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    test_loss /= len(test_loader)
    test_accuracy = correct / total

    print(f"📌 Epoch {epoch} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    if test_loss < best_loss:
        best_loss = test_loss
        epochs_no_improve = 0
        torch.save(pretrained_model.state_dict(), "fine_tuned_without_optimization.pth")
        print(f"✅ New Best Model Saved!")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= early_stop_patience:
            print("⏳ Early stopping triggered.")
            break

print("\n🚀 Training Complete!")
