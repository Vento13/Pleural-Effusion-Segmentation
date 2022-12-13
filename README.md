#  Semantic segmentation baseline with PyTorch for detection of pleural effusion

This repository contains:
- U-Net model architecture
- Pipeline of training and testing model with various losses: Dice loss, BCE_Dice loss, Tversky loss, Focal-Tversky loss
- File with data preprocessing and loading
- Dice coefficient counting function
- subset and subset_mask -- data (images and masks)
- Folder "output": .pt files with models weights and Dice coefficient plots with each epoch
- Folder "runs" with Tensorboard logs
