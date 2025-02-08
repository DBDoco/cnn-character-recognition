<div align="center">

# Handwritten Character and Word Recognition with Genetic Algorithm Optimization

![UI Demo](https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExNjB5NWNlcHB3Mmc5cDQzcjA2Zzlhbm95eHFicGxmbDVybnhjYzJheCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/qXUHudwVJ7UMHBVwhT/giphy.gif)

</div>

## Project Overview

This project implements machine learning models for recognizing handwritten letters, numbers, and words using Convolutional Neural Networks (CNN) and ResNet18 architectures. The models are optimized using a genetic algorithm to improve performance and efficiency.

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

In the `process-image.py` script, you can select between different models:

```python
# For optimized CNN model
model.load_state_dict(torch.load("./models/optimized_character_recognition.pth", map_location=device))

# For unoptimized CNN model
# model.load_state_dict(torch.load("./models/unoptimized_character_recognition.pth", map_location=device))

# For fine-tuned ResNet18 model
# model.load_state_dict(torch.load("./models/finetuned_resnet18.pth", map_location=device))

# For fine-tuned ResNet18 model with genetic algorithm optimization
# model.load_state_dict(torch.load("./models/finetuned_resnet18_optimized.pth", map_location=device))
```

### Running the Application

```bash
python process-image.py
```

1. Launch the application
2. Drag and drop an image of a handwritten character or word
3. The application will predict and display the recognized character or word

## Additional Resources

- [How it works (in Croatian)](https://mega.nz/file/0RJ11LLD#qOdgFtksgz-vKfajwEmmeRb6TghnjmNuMUxsL4VZE6Q)

## Model Comparisons

### CNN Model
- **Without Optimization**: Achieved 85.65% accuracy with 32 epochs.
- **With Genetic Algorithm Optimization**: Achieved 86.13% accuracy with 18 epochs.

### ResNet18 Model
- **Stock ResNet18 (No Fine-Tuning)**: Achieved 2.13% accuracy on EMNIST dataset.
- **Fine-Tuned ResNet18**: Achieved 88.78% accuracy with 37 epochs.
- **Fine-Tuned ResNet18 with Genetic Algorithm Optimization**: Achieved 89.46% accuracy with 24 epochs.

### Handwritten Word Recognition
- **ResNet18 Fine-Tuned on Handwritten Names Dataset**: Achieved 73.85% accuracy, with precision and recall around 33%.

## Fine-Tuning Scripts

Due to the large size of the trained models, we provide the fine-tuning scripts for ResNet18

## Conclusion

This project demonstrates the effectiveness of combining deep learning models with genetic algorithms for optimizing handwritten character and word recognition. The ResNet18 model, especially when fine-tuned and optimized, shows superior performance compared to the basic CNN model. However, challenges remain in recognizing handwritten words due to the variability in handwriting styles and the complexity of the task.

For further improvements, consider balancing the dataset, using more advanced data augmentation techniques, and fine-tuning the model with additional computational resources.
