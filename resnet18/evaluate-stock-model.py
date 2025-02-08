import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_emnist_mapping():
    mapping_path = "../emnist-balanced-mapping.txt"
    mapping = np.loadtxt(mapping_path, delimiter=" ", dtype=int)
    return {i: chr(mapping[i][1]) for i in range(len(mapping))}

label_mapping = load_emnist_mapping()
actual_labels = [label_mapping[i] for i in range(47)]
print("✅ Loaded EMNIST Label Mapping")

pretrained_model = models.resnet18(pretrained=True)

pretrained_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

num_ftrs = pretrained_model.fc.in_features
pretrained_model.fc = nn.Linear(num_ftrs, 47)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_model = pretrained_model.to(device)
pretrained_model.eval()
print("✅ Loaded and modified pretrained model")

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = torchvision.datasets.EMNIST(
    root="./data",
    split="balanced",
    train=False,
    download=True,
    transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = pretrained_model(images)
        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

test_accuracy = accuracy_score(y_true, y_pred)
test_precision = precision_score(y_true, y_pred, average='weighted')
test_recall = recall_score(y_true, y_pred, average='weighted')
test_f1 = f1_score(y_true, y_pred, average='weighted')

print("\n📊 Model Evaluation Results:")
print(f"✅ Accuracy: {test_accuracy:.4f}")
print(f"✅ Precision: {test_precision:.4f}")
print(f"✅ Recall: {test_recall:.4f}")
print(f"✅ F1-score: {test_f1:.4f}")

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", xticklabels=actual_labels, yticklabels=actual_labels)
plt.xlabel("Predicted Characters")
plt.ylabel("True Characters")
plt.title("Confusion Matrix")
plt.show()

print("✅ Evaluation Completed!")
