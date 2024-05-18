# cs577-final-project

**Efficient Image Poisoning as Defense:** Disrupting Profile Matching on OSNs and Preserving Human Comprehension

This repository contains the replication package of our project in two parts; Gaussian Blur Obfuscation, and Salient Feature Obfuscation. It also includes our trial of updating the study "FaceOff", which is a failed attempt.

This project is our final project for the **Bilkent CS 577: Data Privacy** Course. We extend our deepest gratitude to Assistant Professor Sinem Sav for her invaluable guidance throughout this project.

**Group 3, Members**:

- Ecem İlgün
- Mehmet Kadri Gofralilar
- Kousar Kousar
- Noor Muhammad
- Aqsa Shabbir

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [GaussianBlurObfuscation](#gaussianblurobfuscation)
  - [SalientFeatureObfuscation](#salientfeatureobfuscation)
  - [FaceOffReplication](#faceoffreplication)
- [Dataset](#dataset)
- [Results](#results)
  - [Gaussian Blur Obfuscation](#gaussian-blur-obfuscation)
  - [Salient Feature Obfuscation](#salient-feature-obfuscation)

## Installation

For the face_recognition function to work, install Cmake version 3.29.3 from 'https://cmake.org/download/'.

Using python version 3.12.3, you need to install the following dependencies in Python3 for this project.

```bash
pip3 install opencv-python numpy matplotlib mtcnn keras-facenet tensorflow scipy scikit-learn face_recognition
```

## Usage

### GaussianBlurObfuscation

This module is used for applying Gaussian Blur obfuscation to the whole face and calculating the accuracy.

Steps:

1. Just open the file in VSCode or any editor of your choice.
2. Select the python jupyter notebook kernal of choice.
3. Run the cells one by one
<!--

````bash
# Apply
python3 gaussian_blur_bbox_obfuscation/
``` -->

### SalientFeatureObfuscation
This module is used for downloading images from the FaceScrub dataset, applying darkening obfuscation to salient features and the whole face, and calculating the metrics.

```bash
# Download Images, Apply Obfuscation and Print Metrics
python3 salient_feature_obfuscation/main.py
````

Saving metrics is optional and should be provided if wanted as follows.

```bash
# Download Images, Apply Obfuscation and Save Metrics in a File
python3 salient_feature_obfuscation/main.py -m True
```

Two other setups with optional parameters are given as below:

```bash
# Download {x} Images per Person, Apply Obfuscation (default value is 3)
python3 salient_feature_obfuscation/main.py -ni {number of images to download per person}

# Download Images, Apply Obfuscation to {x} of Them Randomly (default value is 200)
python3 salient_feature_obfuscation/main.py -n True -c {number of random people to apply obfuscation}
```

You can check `--help` for details of these parameters.

### FaceOffReplication

This module is an updated version of FaceOff repository originally forked from "https://github.com/wi-pi/face-off". There are still errors, but the usage is still the same as original FaceOff. You may try to run and debug following the instructions given in the original repository.

### Dataset

#### Data Collection

Our code for collecting the data FaceScrub dataset is included in `salient_feature_obfuscation/main.py` file. As for the LFW dataset, ... These files contain scripts and instructions for gathering and preprocessing the dataset used in our experiments. Please refer to these files for detailed steps on how to collect and prepare the data.

#### Overview

The datasets used in our project consists of face images. FaceScrub also includes border locations of the faces.

#### Example Images

##### LFW

![Example Image from LFW](/figs/Alfredo_di_Stefano_0001.jpg)

##### FaceScrub

![Example Image from FaceScrub](/figs/Original_Image_FaceScrub.jpg)

### Results

#### Gaussian Blur Obfuscation

- **Total Number of images**: 618
- **Data set used**: DeepFunneled

##### Example Result Image

![Gaussian Blur example 1](/figs/gaussian_blur_2.jpg)

##### Accuracy

![Gaussian Blur example 1](/figs/gaussian_blur_result.png)

#### Salient Feature Obfuscation

- **Number of Chosen People**: 200
- **Number of Images per Person**: 3

##### Example Result Images

![Applied Salient Feature Obfuscation](/figs/Salient_Image_FaceScrub_1.jpg)
![Applied Whole Face Obfuscation](/figs/Salient_Image_FaceScrub_2.jpg)

##### Accuracy

![Accuracy Results](/figs/Accuracy_FaceScrub.png)

#### Observations

- The 5% approximate decrease in accuracy shows the potential of the boundary box Gaussian blur in evading recognition models. But this result is still not much reliable due to the small sample size used for training and testing the model.

- Our proposed salient feature obfuscation increased the accuracy, which means the matching rate is higher and the accuracy is lower compared to no obfuscation, therefore needs improvements in future.
