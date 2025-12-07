#Load images and creates training and validation sets

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

def resolve_data_dir():
    #Local path in repo
    default_path = "./data/archive/raw-img"

    #Path used in Google Colab if dataset stored in /content
    colab_path = "/content/data/archive/raw-img"

    #Switch to Colab path if running in Colab
    if "/content" in os.getcwd() and os.path.exists(colab_path):
        return colab_path

    return default_path

def get_dataloaders(batch_size=32, debug=False):
    data_dir = resolve_data_dir()

    #Image transforms (resizing + convert to tensor)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5), #augmenting data
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ])

    #Load dataset from folders
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    #Debug mode using small subset of images
    if debug:
        dataset = Subset(dataset, range(min(50, len(dataset))))

    #Split into train and validation sets (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    #Debug mode shrinking both sets
    if debug:
        train_ds = Subset(train_ds, range(min(40, len(train_ds))))
        val_ds = Subset(val_ds, range(min(10, len(val_ds))))

    #Create data laoders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader