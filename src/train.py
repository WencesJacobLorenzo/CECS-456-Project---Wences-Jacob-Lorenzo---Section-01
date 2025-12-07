#Full training script for CNN

import os
import torch
import torch.nn as nn
import torch.optim as optim

from load_dataset import get_dataloaders
from model import build_model
from utils import save_training_curves

EPOCHS = 10
BATCH_SIZE = 32
LR = 0.001
DEBUG = False #Set True for fast debugging

#Helper function for unwrapping nested Subsets
def get_base_dataset(d):
    while hasattr(d, "dataset"):
        d = d.dataset
    return d

def main():
    #Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #Load Data
    train_loader, val_loader = get_dataloaders(batch_size=BATCH_SIZE, debug=DEBUG)

    #Count number of classes
    base_dataset = get_base_dataset(train_loader.dataset)
    num_classes = len(base_dataset.classes)

    #Build model
    model = build_model(num_classes).to(device)

    #Loss + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_losses = []
    val_losses = []
    val_accuracies = []

    #Fast debug mode. Stop after training 1 batch.
    if DEBUG:
        model.train()
        images, labels = next(iter(train_loader))
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        return

    #Main Training Loop
    for _ in range(EPOCHS):
        model.train()
        running_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        #Validation step
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(correct / total)

    #Save model + curves
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/best_model.pth")

    os.makedirs("outputs", exist_ok=True)
    save_training_curves(train_losses, val_losses, val_accuracies)

if __name__ == "__main__":
    main()



