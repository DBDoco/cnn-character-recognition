import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def load_emnist_mapping():
    mapping_path = "../emnist-balanced-mapping.txt"
    mapping = np.loadtxt(mapping_path, delimiter=" ", dtype=int)
    return {i: chr(mapping[i][1]) for i in range(len(mapping))}

label_mapping = load_emnist_mapping()
print("✅ Loaded EMNIST Label Mapping:", label_mapping)

pretrained_model = models.resnet18(pretrained=True)

pretrained_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

num_ftrs = pretrained_model.fc.in_features
pretrained_model.fc = nn.Linear(num_ftrs, 47)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_model = pretrained_model.to(device)
pretrained_model.eval()
print("✅ Loaded and modified pretrained model.")

def process_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),       
        transforms.Resize((28, 28)),   
        transforms.ToTensor(),         
        transforms.Normalize((0.5,), (0.5,)) 
    ])

    image = Image.open(image_path).convert("L")  
    image = transform(image).unsqueeze(0).to(device)  
    return image

def predict_character(image_path):
    image = process_image(image_path)

    with torch.no_grad():
        output = pretrained_model(image)
        predicted_index = torch.argmax(output, dim=1).item()
        predicted_character = label_mapping.get(predicted_index, "?")  

    return predicted_character

if __name__ == "__main__":
    image_path = "24.png" 
    predicted_char = predict_character(image_path)
    print(f"\n🔍 Predicted Character: {predicted_char}")
