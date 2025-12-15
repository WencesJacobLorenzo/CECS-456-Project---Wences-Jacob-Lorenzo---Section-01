# CECS-456-Project---Wences-Jacob-Lorenzo---Section-01

## Chosen Project: [Project Option 4- 10 Animals](https://www.kaggle.com/datasets/alessiocorrado99/animals10)

**Important Note About the Dataset**  
This repository does not include the images in the dataset due to file size limits. However, the required folder structure has been recreated using placeholder files so the code runs correctly after adding the real dataset.

## CNN for Animal Classification
This project impkements a Convolutional Neural Network (CNN) fir 10-class animal image classification with the above Anials-10 dataset. The model is trained using a supervised learning approach with data augmentation, regularization, and adaptive learning-rate scheduling. Evaluation includes both quantitatie and qualitative assessment.

## Requirements
- This project requries Python 3.8 or higher with dependencies listed in **requirements.txt**
- Install dependencies using:
  ```
  pip install -r requirements.txt
  ```

## Running the Project
Before running the training or evaluation scripts, the data must be set up correctly.

### Dataset Preparation
 1. Delete the existing data/archive/ folder (this only contains placeholders).
 2. Download the dataset from the link above.
 3. Replace the dataset's archive/ folder inside the data/ directory, matching this structure **MAKE SURE ARCHIVE IS UNZIPPED**:
   
```
data/
 archive/
  raw-img/
   <category1>/
   <category2>/
   <category3>/
   ...
```
Once the dataset is placed in this structure, you may proceed with training the model from scratch or running evaluation using the provided **best_model.pth** checkpoint.

### Training the Model from Scratch
To train the CNN from scratch and generate a new model checkpoint, run:
```
python train.py
```

During training:
- The model is trained for 30 epochs using the training dataset
- Validation accuracy and loss is evaluated after each epoch
- The best-performing model checkpoint is saved to **models/best_model.pth**
- Training and validation curves are saved to *outputs/training_curves.png**
- This process may take several minutes depending on available hardware. Using GPU runtime in Google Colab can be halpful.

**NOTE:** If you decide to train the model from scratch, you may optionally delete the existing **models/best_model.pth** file before running **train.py** to avoid any confusion with previously saved checkpoints. The training script will automatically save a new best-performing model.

### Evaluation with Pretrained Model
If you want to evaluate the model without retraining, you can run the evaluation with the given checkpoint:
```
python evaluate.py
```

This script:
- Loads the saved model from models/best_model.pth
- Perform inference on the validation dataset
- Print the verall validation accuracy
- Generate evaluation outputs including **outputs/confusion_matrix.png** and **outputs/first_predictions_grid.png**
- This allows for evaluation to be done without redoing the entire training process


