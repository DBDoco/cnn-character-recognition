# Handwritten Character Recognition with Genetic Algorithm Optimization

![UI Demo](https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExNjB5NWNlcHB3Mmc5cDQzcjA2Zzlhbm95eHFicGxmbDVybnhjYzJheCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/qXUHudwVJ7UMHBVwhT/giphy.gif)

## Project Overview

This project implements a machine learning model for recognizing handwritten letters and numbers using a Convolutional Neural Network (CNN) that has been optimized with a genetic algorithm.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, but not required)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/handwritten-character-recognition.git
cd handwritten-character-recognition
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Requirements

The `requirements.txt` file includes the following dependencies:
- torch
- torchvision
- numpy
- matplotlib
- seaborn
- scikit-learn
- deap
- tqdm
- pillow
- opencv-python
- tkinterdnd2

## Usage

### Model Selection

In the `process-image.py` script, you can select between the optimized and unoptimized models:

```python
# For optimized model
model.load_state_dict(torch.load("./models/optimized_character_recognition.pth", map_location=device))

# For unoptimized model
# model.load_state_dict(torch.load("./models/unoptimized_character_recognition.pth", map_location=device))
```

### Running the Application

```bash
python process-image.py
```

1. Launch the application
2. Drag and drop an image of a handwritten character
3. The application will predict and display the recognized character

## Additional Resources

- [How it works (in Croatian)](https://mega.nz/file/gF4GSabb#sJkRNKH3YCFu9SFvpD_nAmNNjmewLctqueYfUXr-rD4)