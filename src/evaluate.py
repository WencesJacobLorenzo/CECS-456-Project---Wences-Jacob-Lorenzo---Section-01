# Load trained model and compute confusion matrix

import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from load_dataset import get_dataloaders
from model import build_model

# Italian to English label translation
translate_labels = {
    "cane": "dog",
    "gatto": "cat",
    "mucca": "cow",
    "cavallo": "horse",
    "elefante": "elephant",
    "gallina": "hen",
    "pecora": "sheep",
    "farfalla": "butterfly",
    "ragno": "spider",
    "scoiattolo": "squirrel"
}

def get_base_dataset(d):
    # Unwrap Subset objects until reaching ImageFolder
    while hasattr(d, "dataset"):
        d = d.dataset
    return d


# Denormalize for image display
def denormalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    img = img.clone().cpu()
    for i in range(3):
        img[i] = img[i] * std[i] + mean[i]
    return torch.clamp(img, 0, 1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load validation data
    _, val_loader = get_dataloaders(batch_size=32, debug=False)

    # Get true number of classes
    base_dataset = get_base_dataset(val_loader.dataset)
    orig_names = base_dataset.classes
    class_names = [translate_labels[name] for name in orig_names]
    num_classes = len(class_names)

    # Build model
    model = build_model(num_classes).to(device)

    # Load trained weights
    model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
    model.eval()

    all_preds = []
    all_labels = []
    first_images = []
    first_true = []
    first_pred = []

    # Collect predictions
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # Save all predictions for confusion matrix
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Save first 5 examples
            if len(first_images) < 5:
                take = 5 - len(first_images)
                first_images.extend(images[:take].cpu())
                first_true.extend(labels[:take].cpu().numpy())
                first_pred.extend(preds[:take].cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Print accuracy
    accuracy = sum(int(p == t) for p, t in zip(all_preds, all_labels)) / len(all_labels)
    print(f"\nValidation Accuracy: {accuracy:.4f}\n")

    # Print first 5 predictions
    print("First 5 Predictions:")
    for i in range(len(first_images)):
        print(f"Image {i+1}: True = {class_names[first_true[i]]}, Pred = {class_names[first_pred[i]]}")

    # Save confusion matrix
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        cmap="Blues",
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    # Save a denormalized visualization of the first 5 images
    plt.figure(figsize=(15, 4))
    for i in range(len(first_images)):
        plt.subplot(1, 5, i+1)

        img = denormalize(first_images[i]).permute(1, 2, 0).numpy()
        plt.imshow(img)

        plt.title(f"T:{class_names[first_true[i]]}\nP:{class_names[first_pred[i]]}")
        plt.axis("off")

    plt.savefig("outputs/first5_predictions.png")
    plt.close()


if __name__ == "__main__":
    main()
