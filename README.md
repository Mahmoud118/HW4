# Comparative Analysis of GAN Models on the CIFAR-10 Dataset

## Brief Description

This project implements Generative Adversarial Networks (GANs) to generate high-quality images on the CIFAR-10 dataset. It incorporates techniques from DCGAN (Deep Convolutional GAN), WGAN (Wasserstein GAN), and ACGAN (Auxiliary Classifier GAN) to compare their performance. The project evaluates the models using discriminator scores, the Inception Score, and visualizes the top 10 generated images.

## Key Features

- Implements DCGAN, WGAN, and ACGAN for image generation.
- Utilizes CIFAR-10 dataset for training.
- Hyperparameter optimization for stable training.
- Inception Score calculation for quantitative evaluation.
- Visualization of discriminator score distribution.
- Saves and ranks the top 10 generated images.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- seaborn
- tqdm
- numpy
- scipy
- PIL (Pillow)

## Installation

Clone the repository:
   ```bash
   git clone https://github.com/Mahmoud118/HW4
   ```

## Model Architectures

### DCGAN
A standard GAN architecture with convolutional layers and BatchNorm. It uses a discriminator and a generator with ReLU/Tanh activations.

### WGAN
An improved GAN architecture employing weight clipping and Wasserstein loss for stable training.

### ACGAN
A GAN that incorporates class labels into the training process, enhancing image generation with auxiliary classification.

## Training

The training script includes:
- Data normalization and augmentation.
- Adaptive learning rates for stability.
- Logging of training losses and scores.
- Saving the best model weights for each architecture.

## Evaluation

- **Top Generated Images**: The top 10 images ranked by discriminator scores are saved.
- **Inception Score (IS)**: Evaluates the diversity and quality of generated images.
- **Discriminator Score Distribution**: Visualizes the scores assigned by the discriminator to generated images.

## Hyperparameters

| Parameter    | Value        |
|--------------|--------------|
| Batch Size   | 128          |
| Epochs       | 60           |
| Learning Rate| 0.0002       |
| Beta1        | 0.5          |
| Z-Dim        | 100          |
| Image Size   | 32x32        |
| Channels     | 3 (RGB)      |

## Visualization

1. **Generated Images**:
   - View the top 10 generated images with their discriminator scores.
2. **Discriminator Score Distribution**:
   - Histogram of scores for all generated images.