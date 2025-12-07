#Full training script for CNN

import os
import torch
import torch.nn as nn
import torch.optim as optim

from load_dataset import get_dataloaders
from model import build_model
from utils import save_training_curves

EPOCHS = 15 #Update epochs
BATCH_SIZE = 32
LR = 0.001
DEBUG = False  # Set True for fast debugging

# Helper function unwrap nested Subsets to reach ImageFolder
def get_base_dataset(d):
    while hasattr(d, "dataset"):
        d = d.dataset
    return d

def main():
    #Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #Load training & validation sets
    train_loader, val_loader = get_dataloaders(batch_size=BATCH_SIZE, debug=DEBUG)

    #Count number of classes
    base_dataset = get_base_dataset(train_loader.dataset)
    num_classes = len(base_dataset.classes)

    # Show that train/val split is consistent
    print("First 10 indices of train_ds:",
          train_loader.dataset.indices[:10])
    print("First 10 indices of val_ds:",
          val_loader.dataset.indices[:10])

    #Build CNN model
    model = build_model(num_classes).to(device)

    #Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    #Lists to store curves
    train_losses = []
    val_losses = []
    val_accuracies = []

    #DEBUG MODE uses 1 training step
    if DEBUG:
        print("DEBUG MODE: Running one quick batch...")
        model.train()

        images, labels = next(iter(train_loader))
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print("Debug step finished successfully.")
        return

    best_acc = 0.0

    #Training Loop
    for epoch in range(EPOCHS):

        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        #Validation Loop
        model.eval()
        val_loss = 0.0
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

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        #Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/best_model.pth")


    #Save training curves
    os.makedirs("outputs", exist_ok=True)
    save_training_curves(train_losses, val_losses, val_accuracies)


if __name__ == "__main__":
    main()


