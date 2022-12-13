import torch

def dice_coef(masks, outputs):
    smooth = 1.0
    outputs = outputs[:, 0].contiguous().view(-1)
    masks = masks[:, 0].contiguous().view(-1)
    intersection = (outputs * masks).sum()
    dsc = (2. * intersection + smooth) / (outputs.sum() + masks.sum() + smooth)
    return dsc