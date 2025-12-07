#Save training curves to image

import matplotlib.pyplot as plt

def save_training_curves(train_loss, val_loss, val_acc, out_path="outputs/training_curves.png"):
    plt.figure(figsize=(10,4))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_acc, label="Val Accuracy")
    plt.legend()

    plt.savefig(out_path)
    plt.close()