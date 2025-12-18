# CNN Model Definition (Improved with Global Average Pooling)

import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Feature extractor
        self.features = nn.Sequential(
            # First block extracts low-level visual features and reduces spatial resolution
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Second block learns mid-level patterns with increased channel depth and further downsampling
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Third block captures higher-level shape and texture combinations
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Fourth block learns abstract, distinct visual representations
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Classifier with global average pooling
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, num_classes)
        )

    # Forward pass
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def build_model(num_classes):
    return SimpleCNN(num_classes)
