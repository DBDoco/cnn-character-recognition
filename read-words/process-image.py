import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image, ImageDraw, ImageFont

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("model.pth", map_location=device)
label2idx = checkpoint["label2idx"]
idx2label = {v: k for k, v in label2idx.items()} 

num_classes = len(label2idx)
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

input_dir = "./images"
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)  

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  
        image_path = os.path.join(input_dir, filename)
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            _, predicted_idx = torch.max(output, 1)
            predicted_label = idx2label[predicted_idx.item()]

        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        image_width, image_height = image.size
        text_bbox = draw.textbbox((0, 0), predicted_label, font=font)  
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_x = image_width - text_width - 10  
        text_y = image_height - text_height - 10  

        draw.text((text_x, text_y), predicted_label, fill="red", font=font)

        output_path = os.path.join(output_dir, filename)
        image.save(output_path)
        print(f"Processed {filename} -> {output_path} (Prediction: {predicted_label})")

print("Processing complete. Annotated images saved in ./output/")
