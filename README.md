# Building Medical Imaging Tools with Deep Learning

Implement CNN architectures and segmentation models that form the core principles of state-of-the-art medical imaging systems. While modern clinical systems use large NNs on massive medical images, I used miniature 2D histopathological images (i.e. PathMNIST). These same principles—convolutional feature extraction and spatial grounding—directly translate to real-world medical imaging applications.

Goal is to use deep learning fundamentals to enable clinical AI systems (not achieving peak performance through automated solutions). The secondary goal is to learn to use LLMs to make your life easier.

### Code Structure Overview
- `classification_main.py`: Training orchestration (provided)
- `classification_models.py`: **TODO** - Implement your classifier neural networks, including a simple MLP classifier and CNNs
- `classification_train.py`: **TODO** - Implement training/validation loops with PyTorch
- `classification_dataset.py`: PathMNIST data loading
- `segmentation_models.py`: **TODO** - Implement segmentation models, including a linear model, a CNN, and a U-Net with skip connections
- `segmentation_train.py`: **TODO** - Implement segmentation training pipeline
- `segmentation_main.py`: Segmentation training orchestration
- `synthetic_data.py`: Synthetic box generation for segmentation

## Part 1: CNN Implementation for Medical Image Classification 

Implement and compare different architectures for PathMNIST tissue classification:

0. **Simple Linear classifier**: Flatten input images and re-implement logistic regression in PyTorch.
1. **MLP**: Flatten input images and use a multi-layer perceptron (MLP) model
2. **My CNN**: Designn and implement a convolutional architecture

**Performance Expectations (1 epoch, 100 steps):**
- MLP Model: ~32% validation accuracy
- My CNN: >54% validation accuracy

**My Final Model Requirements:**
- Train my best model on a longer training schedule. 
- **Target**: >90% validation accuracy

## Part 2: Segmentation Implementation

Implement and compare different architectures for segmentation on PathMNIST:

0. **Simple Linear model**: Flatten input, predict flattened output
1. **MLP**: Flatten input, predict flattened output with an MLP
2. **Simple CNN**: Encoder-decoder without skip connections  
3. **U-Net**: Full U-Net with skip connections

**Performance Expectations (1 epoch, 100 steps):**
- MLP: ~43% validation IoU
- Simple CNN: >94% validation IoU  
- U-Net: >96% validation IoU

**Final Model Requirements:**
- Train my best model on a longer training schedule. 
- **Target**: >97% validation IoU
