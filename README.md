# Deep-Learning-with-PyTorch-ImageSegmentation

# Image Segmentation with PyTorch

## Overview
This project demonstrates how to perform image segmentation using PyTorch. It leverages the `segmentation-models-pytorch` library and `albumentations` for efficient and effective image augmentation. The project includes instructions to set up a Colab environment, install necessary dependencies, and run segmentation models.

## Features
- Image segmentation using pre-trained models
- Custom data augmentation using `albumentations`
- Easy setup in Google Colab with GPU support

## Setup

### Colab Environment
To run this project in Google Colab, follow these steps:

1. **Set up GPU Runtime:**
   - Navigate to `Runtime` > `Change runtime type`
   - Select `GPU` as the hardware accelerator

2. **Install Dependencies:**
   Run the following commands to install the required libraries:
   ```python
   !pip install segmentation-models-pytorch
   !pip install -U git+https://github.com/albumentations-team/albumentations
   !pip install --upgrade opencv-contrib-python
