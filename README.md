# PNeumonia-Classification-with-CNN
Creating a Model that can diagnose pneumonia with an xray image

## Background
Pneumonia is a serious lung infection, often caused by bacteria or viruses. It affects millions worldwide, particularly children and the elderly. Key facts:

##### Prevalence 
Major  contributor to global mortality, especially in young children(It accounts for ~15% of all deaths in children under 5 years old globally according to WHO)
##### Symptoms
Fever, cough, chest pain, and difficulty breathing.
##### Diagnosis and Challenges
Accurate pneumonia diagnosis from chest radiographs (CXRs),which typically appears as increased opacity on chest radiographs (CXRs) is complex requiring expert assessment and consideration of factors like patient positioning and inspiration depth. This task places a regular high-volume demand on clinicians.
##### Machine Learning Significance
Promising tool for early, quicker and precise pneumonia detection.

## Project Dataset
Dataset is from the RSNA Pneumonia challenge on Kaggle . It contains 
1. 26684x-ray images in dicom format.  20672 images without pneumonia and 6012 images with pneumonia. 
2. Training data , a csv file providing patientIDs, bounding boxes and labels.
The x-ray images are named with the corresponding PatientIds.

## Dependencies
pandas
pydicom
numpy
cv2
torch
torchvision
matplotlib
torchmetrics
pytorch_lightning

## Preprocessing 
1. Original image shape (1024 * 1024) was resized to (224 * 224) for simplicity using cv2
2. The pixel values was standardized into [0,1] by scaling with 1/255
4. Dataset was split into 24000 train images and 2684 validation images
5. Processed images are stored in corresponding folders: 0 if no pneumonia and 1 pneumonia
6. Z-normalization of images with computed mean and std was done
7. Data Augmentation applied are Random Rotations, Random Translations, Random Scales and Random resized crops

   Preprocessing was done in a local environment 

   

## Training
1. Network Architecture: ResNet18 model with the following changes
   
      A. Initial Convolutional Layer:
       i. Changed input channels from 3 to 1 since we are working with grayscale images
       ii. output channel is 64
       iii. kernel size is 7*7, with stride (2,2) and padding (3,3)
   
      B. Last Fully Connected Layer : Changed output dimension fromm 1000 to 1 suitable for binary classification
   
3. Loss Function : BCEWithLogitsLoss
      Directly applied to logits (raw prediction)
      Negative output turned into No pneumonia
4. Optimizer: Adam (lr=1e-4)
5. Trained for 35 epochs
   
Training was done in google collaboratory environment to utilize A-100 GPU 

## Evaluation and testing

1.Performance Metrics used for model selection is Validation Epoch Average Loss
2.Validation accuracy lies around 84% which shows this can still be improved
3.Precision is higher than recall with default threshold of 0.5. This is not so good for this scenerio as it is better to have extremely low false negatives i.e we dont want to miss anyone that has pneumonia
4.Using a threshold of 0.25 improved the recall significantly


## Project Limitations and Future Work
Class imbalance affected the model performance as we have excessively larger number of images with no pneumonia than with pneumonia. I will be trying weighted loss and oversampling techniques in a bid to get a better perfomance. 




