# CECS-456-Project---Wences-Jacob-Lorenzo---Section-01

## Chosen Project: [Project Option 4 - 10 Animals](https://www.kaggle.com/datasets/alessiocorrado99/animals10)

**Important Note About the Dataset**  
This repository does not include the images in the dataset due to file size limits. However, the required folder structure has been recreated using placeholder files so the code runs correctly after adding the real dataset.

## CNN for Animal Classification
This project implements a Convolutional Neural Network (CNN) for 10-class animal image classification with the above Animals-10 dataset. The model is trained using a supervised learning approach with data augmentation, regularization, and adaptive learning-rate scheduling. Evaluation includes both quantitative and qualitative assessment.

## Folders
This repository is organized into several folsers to clearly separate the source code, dataset, model outputs, and generated results
- **src/** - Contains all Python source code for the model and training pipeline
  - load_datasets.py - Loads images, applies transformations, and creates training/validation data
  - model.py - Defines the CNN architecture
  - train.py - Full training script running epoch, logs metrics, and saves the best model
  - evaluate.py - Loads the trained model and generates evaluation outputs like confusion matrix and example predictions
  - utils.py - Helper functions that save training curves
    
- **data/** - Contains dataset used for training and validation. **Note:** The repo includes placeholder files only. You must download and insert the real dataset before training. Setup instructions are shown later in README

- **models/** - Stores trained model weights. This folder is created automatically after training if it does not exist.
   - best_model.pth - THe highest-acuracy model checkpoint saved during training.

- **outputs/** - Stores generated visualization and evaluation results. This folder is also auto-created if it does not exist.
  - training_curves.png - Plot of training/valifation loss and accuracy over epochs
  - confusion_matrix.png - Confusion matrix for model predictions on the validation set
  - first_predictions_grid.png - Grid showing sample predictions on the first 20 validation images.

## Requirements
- This project requires Python 3.8 or higher with dependencies listed in **requirements.txt**
- Install dependencies using:
  ```
  pip install -r requirements.txt
  ```

## Running the Project
Before running the training or evaluation scripts, the data must be set up correctly.

### Dataset Preparation
 1. Delete the existing **archive/** folder (this only contains placeholders).
 2. Download the dataset from the link above.
 3. Replace the dataset's **archive/** folder inside the data/ directory, matching this structure:
   
```
data/
 archive/
  raw-img/
   <category1>/
   <category2>/
   <category3>/
   ...
```
  4. Make sure **archive/** folder is unzipped. Unzip if needed.

Once the dataset is placed in this structure and archive/ is unzipped, you may proceed with training the model from scratch or running evaluation using the provided **best_model.pth** checkpoint.

### Training the Model from Scratch
To train the CNN from scratch and generate a new model checkpoint, run:
```
python src/train.py
```

During training:
- The model is trained for 30 epochs using the training dataset
- Validation accuracy and loss is evaluated after each epoch
- The best-performing model checkpoint is saved to **models/best_model.pth**
- Training and validation curves are saved to **outputs/training_curves.png**
- **This process may take a while depending on available hardware. Using GPU runtime in Google Colab can be halpful.**

**NOTE:** If you decide to train the model from scratch, you may optionally delete the existing **models/best_model.pth** file before running **train.py** to avoid any confusion with previously saved checkpoints. The training script will automatically save a new best-performing model.

### Evaluation with Pretrained Model
If you want to evaluate the model without retraining, you can run the evaluation with the given checkpoint:
```
python src/evaluate.py
```

This script:
- Loads the saved model from **models/best_model.pth**
- Perform inference on the validation dataset
- Print the verall validation accuracy
- Generate evaluation outputs including **outputs/confusion_matrix.png** and **outputs/first_predictions_grid.png**
- This allows for evaluation to be done without redoing the entire training process


