# CECS-456-Project---Wences-Jacob-Lorenzo---Section-01

## Chosen Project: [Project Option 4- 10 Animals](https://www.kaggle.com/datasets/alessiocorrado99/animals10)

**Important Note About the Dataset**  
This repository does not include the images in the dataset due to file size limits. However, the required folder structure has been recreated using placeholder files so the code runs correctly after adding the real dataset.

## CNN for Animal Classification
This project impkements a Convolutional Neural Network (CNN) fir 10-class animal image classification with the above Anials-10 dataset. The model is trained using a supervised learning approach with data augmentation, regularization, and adaptive learning-rate scheduling. Evaluation includes both quantitatie and qualitative assessment.

To run the project:
 1. Delete the existing data/archive/ folder (this only contains placeholders).
 2. Download the dataset from the link above.
 3. Replace the dataset's archive/ folder inside the data/ directory, matching this structure:
   
```
data/
 archive/
  raw-img/
   <category1>/
   <category2>/
   <category3>/
   ...
```
