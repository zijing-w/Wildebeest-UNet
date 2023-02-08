# Satellite-based monitoring of the world's largest terrestrial mammal migration using deep learning
This repository contains the neural network model (UNet) pipeline and other essential codes for detecting wildebeest and zebras in Serengeti-Mara ecosystem. Please feel free to contact me at zijingwu97@outlook.com if you have any questions.

## Setup and Installation
The most easy way to test and use this code is to upload the files to Google Drive and open the notebooks with Google Colaboratory.
Alternatively, you can install the required packages (see requirements.txt) on your computer and open the notebooks with Jupyter Notebook.

## Folder structure
### core: 
the necessary modules to run the code.

#### SampleData: 
the sample dataset required to test the code.
##### SampleData/1_Data_preparation: 
sample dataset for data preparation.
##### SampleData/data_2009Aug/3_Train_test: 
a sample training and test dataset
##### SampleData/data_2009Aug/classify: 
a sample satellite image

#### tmp: 
the folder to save temporary weight files while running the code. The pretrained weights for wildebeest detection are also stored here.
##### tmp/checkpoint: 
the folder to save weight files that are generated during model training.
##### tmp/logs: 
the folder to save the training records for the use of tensorboard.
##### tmp/parameter_tuning: 
the folder to save the weights for parameter tuning.
##### tmp/pretrained_weights: 
the pretrained weights for wildebeest detection.

#### 1_Preprocessing_AOI_to_Mask.ipynb: 
the notebook for preprocessing the data.

#### 2_Wildebeest_detection_using_UNet.ipynb: 
the notebook for model training.

#### 3_Postprocessing_ wildebeest_counting.ipynb: 
the notebook for detecting and counting the wildebeest on the image with the trained model.

#### requirements.txt: required libraries and packages to run the notebooks.

## Steps below to run the code

### Step 1: Data preparation - [1_Preprocessing_AOI_to_Mask.ipynb]
The data has two main components, the satellite images and the label of animals in those images (in polygon shapefile). 
The SampleData/1_Data_preparation directory contains examples of the satellite image and the polygon annotations.
The notebook convert the polygon annotations to mask images that will be fed into the U-Net model.

### Step 2: Model training - [2_Wildebeest_detection_using_UNet.ipynb]
Remember to configure the Data path of the training images and masks as well as the test image and masks.
Train the UNet model with the images and labelled masks. A sample dataset is stored in SampleData/data_2009Aug/3_Train_test.
There are also sections of parameter tuning in this notebook.
Modify the number of channels according to your data type (R G B bands or multiple bands).

### Step 3: Postprocessing - [3_Postprocessing_ wildebeest_counting.ipynb]
Use the trained model to detect the animals on the images and count the animals. Make sure that the RAM is sufficient to run the code.
A sample satellite image is stored in SampleData/data_2009Aug/classify and the pretrianed weights can be found in tmp/pretrained_weights.
