import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def load_emnist_mapping():
    mapping_path = "../emnist-balanced-mapping.txt"
    mapping = np.loadtxt(mapping_path, delimiter=" ", dtype=int)
    label_map = {i: chr(mapping[i][1]) for i in range(len(mapping))}
    return label_map

label_map = load_emnist_mapping()
print("✅ Loaded EMNIST Label Mapping")

def load_fine_tuned_model():
    model = torchvision.models.resnet18()

    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.32),
        nn.Linear(num_ftrs, 47)
)

    model.load_state_dict(torch.load("./models/fine_tuned_with_optimization.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    model.eval()
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_fine_tuned_model().to(device)
print("✅ Fine-Tuned ResNet18 Loaded!")

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

batch_size = 64
test_dataset = torchvision.datasets.EMNIST(
    root="./data",
    split="balanced",
    train=False,
    download=True,
    transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

y_true, y_pred = [], []
test_loss = 0.0
correct = 0
total = 0
criterion = nn.CrossEntropyLoss()

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

test_accuracy = accuracy_score(y_true, y_pred)
test_precision = precision_score(y_true, y_pred, average='weighted')
test_recall = recall_score(y_true, y_pred, average='weighted')
test_f1 = f1_score(y_true, y_pred, average='weighted')
test_loss /= len(test_loader)

print("\n📌 Fine-Tuned Model Performance:")
print(f"✅ Accuracy: {test_accuracy:.4f}")
print(f"✅ Precision: {test_precision:.4f}")
print(f"✅ Recall: {test_recall:.4f}")
print(f"✅ F1-score: {test_f1:.4f}")
print(f"✅ Test Loss: {test_loss:.4f}")

actual_labels = [label_map[i] for i in range(47)]
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", xticklabels=actual_labels, yticklabels=actual_labels)
plt.xlabel("Predicted Characters")
plt.ylabel("True Characters")
plt.title("Confusion Matrix")
plt.show()

print("✅ Evaluation Completed!")
