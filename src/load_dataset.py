# Load images and create training / validation sets

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

def resolve_data_dir():
    default_path = "./data/archive/raw-img"
    colab_path = "/content/data/archive/raw-img"

    if "/content" in os.getcwd() and os.path.exists(colab_path):
        return colab_path
    return default_path


def get_dataloaders(batch_size=32, debug=False):
    data_dir = resolve_data_dir()

    # Data augmentation + transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])

    # Load dataset
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    # Debug mode small subset
    if debug:
        dataset = Subset(dataset, range(min(50, len(dataset))))

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # *** FIXED SPLIT: prevents reshuffling ***
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    # Debug small subsets
    if debug:
        train_ds = Subset(train_ds, range(min(40, len(train_ds))))
        val_ds = Subset(val_ds, range(min(10, len(val_ds))))

    # Dataloaders (train shuffles, val does not)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
