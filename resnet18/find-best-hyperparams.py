import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from deap import base, creator, tools, algorithms
import random
import numpy as np
from tqdm import tqdm

def load_emnist_mapping():
    mapping_path = "../emnist-balanced-mapping.txt"
    mapping = np.loadtxt(mapping_path, delimiter=" ", dtype=int)
    label_map = {i: chr(mapping[i][1]) for i in range(len(mapping))}
    return label_map

label_map = load_emnist_mapping()
print("✅ Loaded EMNIST Label Mapping")

def load_fine_tuned_resnet():
    model = models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 47)  # Ensure this matches saved model

    model.load_state_dict(torch.load("./models/fine_tuned_resnet18.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))

    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    return model


def train_and_evaluate(learning_rate, dropout_rate, batch_size):
    batch_size = int(batch_size)

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(10, translate=(0.2, 0.2), shear=10, scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.EMNIST(
        root="./data",
        split="balanced",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.EMNIST(
        root="./data",
        split="balanced",
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_weights = compute_class_weight(
        "balanced",
        classes=np.array(range(47)),
        y=train_dataset.targets.numpy()
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_fine_tuned_resnet().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = 5  
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total  
    print(f"📊 Validation Accuracy: {accuracy:.4f}")
    
    return accuracy  

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("learning_rate", random.uniform, 0.0001, 0.01)
toolbox.register("dropout_rate", random.uniform, 0.1, 0.5)
toolbox.register("batch_size", random.randint, 32, 128)

toolbox.register("individual", tools.initCycle, creator.Individual, 
                 (toolbox.learning_rate, toolbox.dropout_rate, toolbox.batch_size), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def mutate(individual):
    learning_rate, dropout_rate, batch_size = individual
    
    learning_rate = max(0.0001, learning_rate + random.gauss(0, 0.002))  
    dropout_rate = min(0.5, max(0.1, dropout_rate + random.gauss(0, 0.05))) 
    batch_size = max(32, min(128, int(batch_size + random.gauss(0, 8)))) 
    
    individual[:] = [learning_rate, dropout_rate, batch_size] 
    return (individual,)  

toolbox.register("mutate", mutate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", lambda ind: (train_and_evaluate(*ind),))  

def optimize_hyperparameters():
    population = toolbox.population(n=5) 
    ngen = 3 

    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=ngen, verbose=True)

    best_individual = tools.selBest(population, 1)[0]
    print(f"\n✅ Best Hyperparameters Found: Learning Rate={best_individual[0]:.5f}, Dropout={best_individual[1]:.2f}, Batch Size={int(best_individual[2])}")
    return best_individual

best_hyperparams = optimize_hyperparameters()

final_learning_rate, final_dropout_rate, final_batch_size = best_hyperparams
train_and_evaluate(final_learning_rate, final_dropout_rate, final_batch_size)
