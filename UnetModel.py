from torch import nn
import torch

class Unet(nn.Module):

    def __init__(self):
        super().__init__()
                
        self.dconv_down1 = self.double_conv(1, 64)
        self.dconv_down2 = self.double_conv(64, 128)
        self.dconv_down3 = self.double_conv(128, 256)
        self.dconv_down4 = self.double_conv(256, 512)
        self.dconv_down5 = self.double_conv(512, 1024)      

        self.maxpool = nn.MaxPool2d(2)
        self.upsample5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dconv_up5 = self.double_conv(512 + 1024, 512)

        self.upsample4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dconv_up4 = self.double_conv(256 + 512, 256)

        self.upsample3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dconv_up3 = self.double_conv(128 + 256, 128)

        self.upsample2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dconv_up2 = self.double_conv(128 + 64, 64)
        
        self.upsample1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv_last1 = nn.Conv2d(64, 1, 1)

    def double_conv(self, in_channels, out_channels):
      return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=out_channels),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True)
    )   
        
        
    def forward(self, x):

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)  

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        conv5 = self.dconv_down5(x)
        x = self.maxpool(conv5)

        x = self.upsample5(x)
        x = torch.cat((x, conv5), dim=1)
        x = self.dconv_up5(x)
        
        x = self.upsample4(x)
        x = torch.cat((x, conv4), dim=1)
        x = self.dconv_up4(x)

        x = self.upsample3(x)
        x = torch.cat((x, conv3), dim=1)
        x = self.dconv_up3(x)

        x = self.upsample2(x)
        x = torch.cat((x, conv2), dim=1)   
        x = self.dconv_up2(x)
        
        x = self.upsample1(x)
        out = self.conv_last1(x)
        
        return out