# Pneumonia Classification Using Transfer Learning (ResNet18 and ResNet50)
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
   
3. Loss Function : BCEWithLogitsLoss. Using pos_weight 1,2.22 and 4.44 (26684/6012=4.44 to handle class imbalance by giving more weight to the minority positive class)
   Loss Function directly applied to logits (raw prediction)
      
4. Optimizer: Adam (lr=1e-4) with ReduceLROnPlateau scheduler to update learning rate based on Average Validation loss performance
5. Trained for 10 epochs
   
Training was done in google collaboratory environment to utilize V-100 GPU 

## Evaluation and testing

RESNET-18

1.Performance Metrics used are Validation Epoch Average Loss, Accuracy, Precision , Recall and F1-score.

2.Best Validation accuracy lies around 84% which shows this can still be improved

3.Overall Best model across all weights is found at Using Weight 1.0, with validation loss: 0.35049563578583975 at epoch 9

4.Overall Best Accuracy found at Using Weight 1.0, with Accuracy : 84.87% at epoch 4

5.Overall Best Weighted Precision found at Using Weight 4.440000057220459, with Precision : 0.84 at epoch 5

6.Overall Best Weighted Recall found at Using Weight 1.0, with Recall : 0.85 at epoch 4 

7.Overall Best Weighted F1 found at Using Weight 1.0, with F1 : 0.84 at epoch 2

8.As expected, best positive class Recall (0.85) was obtained using weight 4.44 at epoch 4. Since more emphasis is placed on the minority class using pos_weight 
  in the loss function



## Trade-Offs
1.If both classes are prioritized, using weight 1 produced a better model

2.If Positive class is prioritized(i.e picking out pneunomia ,more suitable for diagnosis ). Using model of weight of 4.44  will be better since it has higher recall i.e we dont want to miss anyone that has pneumonia and dont mind many false positives. Reducing the threshold from 0.5 will also help in this direction


## Project Limitations and Future Work
Class imbalance affected the model performance as we have excessively larger number of images with no pneumonia than with pneumonia. 
Weighted Loss helped class imbalance as it improved positive class recall but caused model overfitting on the minority class examples, causing it to perform poorly on the majority class.
I will be trying Oversampling techniques in a bid to get a better perfomance. 


# RESNET-50

Resnet-50 performed slighly lower than RESNET-18 as seen in the 'PneumoniaResnet50_training_pos_weights.ipynb' file




