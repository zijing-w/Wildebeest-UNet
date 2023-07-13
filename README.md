# Deep learning enables satellite-based monitoring of large populations of terrestrial mammals across heterogeneous landscapes
This repository contains the neural network model (UNet) pipeline and other essential codes for detecting wildebeest and zebras in Serengeti-Mara ecosystem from very-high-resolution satellite imagery. Please feel free to contact me at zijingwu97@outlook.com if you have any questions.

Link to the paper: https://www.nature.com/articles/s41467-023-38901-y or https://rdcu.be/dc8bU

## Setup and Installation
The most easy way to test and use this code is to upload all the files to Google Drive and open the notebooks with Google Colaboratory (https://colab.research.google.com/).
Alternatively, you can install the required packages (see requirements.txt) on your computer and open the notebooks with Jupyter Notebook.
Note: if you use Google Colaboratory, you may encounter some issues about deprecated arguments/functions because Colaboratory keeps updating the packages and the latest packages may not be exactly the same as the ones used in the code here. The issues are usually easy to solve though, so don't hesitate and just give it a try!

## Folder structure
### core: 
the necessary modules to run the code.

### SampleData: 
the sample dataset required to test the code.
#### SampleData/1_Data_preparation: 
sample dataset for data preparation: coverting the wildebeest annotations to segmentation masks.
#### SampleData/data_2009Aug: 
a sample training/test dataset of year 2009 for model training/testing.
##### SampleData/data_2009Aug/3_Train_test: 
a sample training and test dataset. Note that we cannot share the samples of satellite images because the satellite images were acquired by the Smithsonian Conservation Biology Institute and the United States Army Research Laboratory under a NextView Imagery End User License Agreement. The copyright remains with Maxar Technologies (formally DigitalGlobe), and redistribution is not possible.
##### SampleData/data_2009Aug/classify: 
a sample satellite image. We have removed the image here because the satellite images were acquired by the Smithsonian Conservation Biology Institute and the United States Army Research Laboratory under a NextView Imagery End User License Agreement. The copyright remains with Maxar Technologies (formally DigitalGlobe), and redistribution is not possible.

### tmp: 
the folder to save temporary weight files while running the code.
#### tmp/checkpoint: 
the folder to save weight files that are generated during model training.
#### tmp/logs: 
the folder to save the training records for the use of tensorboard.
#### tmp/sensitivity_analysis: 
the folder to save the weights for sensitivity analysis.
#### tmp/pretrained_weights: 
the pretrained weights for wildebeest detection. The weights are not stored here because of file size limit. Please contact zijingwu97@outlook.com to request the data.

### 1_Preprocessing_AOI_to_Mask.ipynb: 
the notebook for preprocessing the data.

### 2_Wildebeest_detection_using_UNet.ipynb: 
the notebook for model training.

### 3_Postprocessing_ wildebeest_counting.ipynb: 
the notebook for detecting and counting the wildebeest on the image with the trained model.

### requirements.txt: 
required libraries and packages to run the notebooks.

## Steps to run the code

### Step 1: Data preparation - [1_Preprocessing_AOI_to_Mask.ipynb]
The data has two main components, the satellite images and the label of animals in those images (in the shapefile). 
The SampleData/1_Data_preparation directory contains examples of the satellite image and the annotations.
The notebook convert the annotations to image mask that will be fed into the U-Net model.

### Step 2: Model training - [2_Wildebeest_detection_using_UNet.ipynb]
Remember to configure the Data path of the training images and masks as well as the test image and masks.
Train the UNet model with the images and labelled masks. A sample dataset is stored in SampleData/data_2009Aug/3_Train_test.
Modify the number of channels according to your data type (R G B bands or multiple bands).

### Step 3: Postprocessing - [3_Postprocessing_ wildebeest_counting.ipynb]
Use the trained model to detect the animals on the images and count the animals. Make sure that the RAM is sufficient to run the code.
A sample satellite image is stored in SampleData/data_2009Aug/classify..
