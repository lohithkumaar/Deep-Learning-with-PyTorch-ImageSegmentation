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

## Usage
   **Data Preparation**
   Prepare your dataset with images and corresponding masks. Ensure the dataset structure is as follows:
   ```
   dataset/
   │
   ├── images/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   └── masks/
       ├── mask1.png
       ├── mask2.png
       └── ...
   ```
## Training the Model
 **1. Import necessary libraries and modules:**
 ```
  import segmentation_models_pytorch as smp
  from albumentations import Compose, RandomCrop, HorizontalFlip, Normalize
  from albumentations.pytorch import ToTensorV2
```
 **2. Define the augmentation pipeline:**
 ```
  import segmentation_models_pytorch as smp
  from albumentations import Compose, RandomCrop, HorizontalFlip, Normalize
  from albumentations.pytorch import ToTensorV2
```
**3. Load your data and create data loaders.:**
**4. Define the model, loss function, and optimizer:**
```
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
loss = smp.losses.DiceLoss()
optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001)])
```
**5.Train the model:**
```
num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    for images, masks in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss_value = loss(outputs, masks)
        loss_value.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss_value.item():.4f}")
```
**6.Inference**
After Training, use the trained model to perform inference on new images:
```
model.eval()
with torch.no_grad():
    for image in test_images:
        output = model(image.unsqueeze(0))
        predicted_mask = output.squeeze().cpu().numpy()
        # Post-process the predicted mask as required
```
## Results
 Include some example images and their segmented outputs to showcase the model performance.

## Contributing
  Feel free to submit issues and enhancement requests. Contributions are welcome.

## Acknowlegement
   - [Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch)
   - [Albumentations](https://github.com/albumentations-team/albumentations)
