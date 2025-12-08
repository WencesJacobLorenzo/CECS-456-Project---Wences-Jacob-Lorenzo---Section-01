# load_dataset.py
# Load images and creates training and validation sets

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

    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomResizedCrop(180, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),

        # Normalization
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Validation transform (NO augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((180, 180)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Load dataset without transforms (so transforms apply AFTER split)
    full_dataset = datasets.ImageFolder(data_dir)

    # Debug shrink
    if debug:
        full_dataset = Subset(full_dataset, range(min(50, len(full_dataset))))

    # 80/20 split with fixed seed
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Apply transforms after the split
    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = val_transform

    # Debug shrink
    if debug:
        train_ds = Subset(train_ds, range(min(40, len(train_ds))))
        val_ds = Subset(val_ds, range(min(10, len(val_ds))))

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

