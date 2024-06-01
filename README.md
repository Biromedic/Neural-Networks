# Transfer Learning with Pre-trained ResNet-50 on CIFAR-10 Dataset

## Overview

This project demonstrates the application of transfer learning using a pre-trained ResNet-50 model on the CIFAR-10 dataset. The goal is to classify the images in CIFAR-10 into 10 different classes. The pre-trained ResNet-50 model, trained on ImageNet, is fine-tuned by replacing its final layer and training it on the CIFAR-10 dataset.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 testing images. The 10 classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## Steps

### 1. Data Preparation

The dataset is loaded and preprocessed using the following transformations:

- Resize to 256x256 pixels
- Center crop to 224x224 pixels
- Convert to tensor
- Normalize using ImageNet mean and standard deviation values

The dataset is then split into training, validation, and test sets.

### 2. Model Setup

A pre-trained ResNet-50 model is loaded, and all its layers are frozen to prevent updating during training. The final fully connected layer is replaced to match the number of classes in CIFAR-10.

### 3. Training

The model is trained for 10 epochs with the following settings:

- Optimizer: Adam with a learning rate of 0.001 and weight decay of 1e-4
- Loss Function: Cross-Entropy Loss
- Learning Rate Scheduler: `ReduceLROnPlateau` to reduce the learning rate when the validation loss plateaus

### 4. Evaluation

The model is evaluated on the test set to measure its accuracy. Additionally, the training and validation loss are tracked and plotted to visualize the training process.

### 5. Visualization

The predictions of the model are visualized by displaying some test images along with their predicted and true labels.

## Code

The code for this project is structured as follows:

- **Data Preparation**: Loading and preprocessing the CIFAR-10 dataset.
- **Model Setup**: Loading the pre-trained ResNet-50 model and modifying the final layer.
- **Training**: Training the model with the specified optimizer and scheduler.
- **Evaluation**: Evaluating the model on the test set and plotting the results.
- **Visualization**: Visualizing the model's predictions on some test images.

## Conclusion

This project demonstrates the effectiveness of transfer learning using a pre-trained ResNet-50 model on the CIFAR-10 dataset. By freezing the pre-trained layers and fine-tuning the final layer, the model achieves significant performance with relatively less training time and data.
