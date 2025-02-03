import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import ImageTk

def load_emnist_mapping():
    mapping_path = "./emnist-balanced-mapping.txt"
    mapping = np.loadtxt(mapping_path, delimiter=" ", dtype=int)
    return {i: chr(mapping[i][1]) for i in range(len(mapping))}

label_mapping = load_emnist_mapping()
print("✅ Loaded EMNIST Label Mapping:", label_mapping)

class CharacterRecognitionCNN(nn.Module):
    def __init__(self):
        super(CharacterRecognitionCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 47)  
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CharacterRecognitionCNN().to(device)
model.load_state_dict(torch.load("./models/optimized_character_recognition.pth", map_location=device))
model.eval()

def correct_inverted_image(image):
    image_cv = np.array(image)
    _, thresh = cv2.threshold(image_cv, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    white_pixels = np.count_nonzero(thresh == 255)
    black_pixels = np.count_nonzero(thresh == 0)
    
    if white_pixels > black_pixels:
        print("🔄 Inverted image detected. Fixing contrast...")
        image = ImageOps.invert(image)  
    
    return image

def process_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = Image.open(image_path).convert("L")
    image = correct_inverted_image(image)
    image = transform(image).unsqueeze(0).to(device)
    return image

def predict_character(image_path):
    image = process_image(image_path)
    with torch.no_grad():
        output = model(image)
        predicted_index = torch.argmax(output, dim=1).item()
        predicted_character = label_mapping.get(predicted_index, "?")
    return predicted_character

class CharacterRecognitionApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()

        self.title("Character Recognition")
        self.geometry("400x500")
        self.configure(bg="white")

        self.drop_label = tk.Label(self, text="Drag and drop an image here", bg="white", fg="black", font=("Arial", 14))
        self.drop_label.pack(pady=20)

        self.canvas = tk.Canvas(self, width=200, height=200, bg="white", highlightthickness=1, relief="solid")
        self.canvas.pack(pady=10)

        self.result_label = tk.Label(self, text="Prediction: ?", font=("Arial", 20, "bold"), bg="white", fg="blue")
        self.result_label.pack(pady=20)

        self.drop_target_register(DND_FILES)
        self.dnd_bind("<<Drop>>", self.on_drop)

    def on_drop(self, event):
        file_path = event.data.strip("{}") 
        self.display_image(file_path)
        predicted_char = predict_character(file_path)
        self.result_label.config(text=f"Prediction: {predicted_char}")

    def display_image(self, file_path):
        image = Image.open(file_path)
        image.thumbnail((200, 200))  
        img_tk = ImageTk.PhotoImage(image)
        self.canvas.create_image(100, 100, image=img_tk, anchor="center")
        self.canvas.image = img_tk  

if __name__ == "__main__":
    app = CharacterRecognitionApp()
    app.mainloop()
