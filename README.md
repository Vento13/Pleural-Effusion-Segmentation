#  Semantic segmentation baseline with PyTorch for detection of pleural effusion

Test task for AIDiagnostic

This repository contains:
- U-Net model architecture: `./UnetModel.py`
- Data preprocessing and loading: `./DataPreprocessing.py`
- Dice coefficient counting function: `./DiceCoefficient.py`
- Pipeline of training and testing model with various losses (BCE_Dice loss, Dice loss, Tversky loss, Focal-Tversky loss): `./main.ipynb`
- Data: `./subset/`, `./subset_masks/`
- .pt files with models weights and Dice coefficient plots with each epoch: `./output/`
- Tensorboard logs: `./runs/`

---

### Results
| Loss function | Dice coefficient |
| --- | --- |
| BCE-Dice |![Dice coefficient for U-Net with BCE-Dice loss](https://github.com/Vento13/Pleural-Effusion-Segmentation/blob/main/output/DiceCoef_Unet_with_BCE-Dice_loss.png?raw=true) |
| Dice | ![Dice coefficient for U-Net with Dice loss](https://github.com/Vento13/Pleural-Effusion-Segmentation/blob/main/output/DiceCoef_Unet_with_Dice_Loss.png?raw=true) |
| Tversky | ![Dice coefficient for U-Net with Tversky loss](https://github.com/Vento13/Pleural-Effusion-Segmentation/blob/main/output/DiceCoef_Unet_with_Tversky_loss.png?raw=true)
| Focal-Tversky | ![Dice coefficient for U-Net with Focal-Tversky loss](https://github.com/Vento13/Pleural-Effusion-Segmentation/blob/main/output/DiceCoef_Unet_with_FocalTversky_loss.png?raw=true) |
