#  Semantic segmentation baseline with PyTorch for detection of pleural effusion

Test task for AIDiagnostic

This repository contains:
- U-Net model architecture
- Pipeline of training and testing model with various losses: Dice loss, BCE_Dice loss, Tversky loss, Focal-Tversky loss
- File with data preprocessing and loading
- Dice coefficient counting function
- subset and subset_mask – data (images and masks)
- Folder "output": .pt files with models weights and Dice coefficient plots with each epoch
- Folder "runs" with Tensorboard logs

### Results
| Loss function | Dice coefficient |
| --- | --- |
| BCE-Dice |![Dice coefficient for U-Net with BCE-Dice loss](https://github.com/Vento13/Pleural-Effusion-Segmentation/blob/main/output/DiceCoef_Unet_with_BCE-Dice_loss.png?raw=true) |
| Dice | ![Dice coefficient for U-Net with Dice loss](https://github.com/Vento13/Pleural-Effusion-Segmentation/blob/main/output/DiceCoef_Unet_with_Dice_Loss.png?raw=true) |
| Tversky | ![Dice coefficient for U-Net with Tversky loss](https://github.com/Vento13/Pleural-Effusion-Segmentation/blob/main/output/DiceCoef_Unet_with_Tversky_loss.png?raw=true)
| Focal-Tversky | ![Dice coefficient for U-Net with Focal-Tversky loss](https://github.com/Vento13/Pleural-Effusion-Segmentation/blob/main/output/DiceCoef_Unet_with_FocalTversky_loss.png?raw=true) |
