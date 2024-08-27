import torch
import torch.nn as nn
import torch.nn.functional as F
from initial_block import Initial

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ratio, dropout_rate):
        super(UpsampleBlock, self).__init__()
        self.dropout_rate = dropout_rate
        self.inter_channels = in_channels//ratio

        #On main path for dimensionality reduction
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        #PROJECTION 
        self.projection = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, stride=1)

        #CONV
        self.conv2 = nn.ConvTranspose2d(self.inter_channels, self.inter_channels, kernel_size=2, stride=2)

        #EXPANSION
        self.expansion = nn.Conv2d(self.inter_channels, out_channels, 1, 1)

        self.bn1 = nn.BatchNorm2d(self.inter_channels)
        self.bn2 = nn.BatchNorm2d(self.inter_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()
        self.regularizer = nn.Dropout2d(dropout_rate)

    def forward(self, x): 
        #main path
        identity = self.upsample(x)
        identity = self.bn3(self.prelu(self.conv1(identity)))
        

        #right path
        x_r = self.bn1(self.prelu(self.projection(x)))
        x_r = self.bn2(self.prelu(self.conv2(x_r)))
        x_r = self.bn3(self.prelu(self.expansion(x_r)))
        x_r = self.regularizer(x_r)

        return x_r + identity


if __name__ == "__main__":
    x = torch.randn((1, 128, 64, 64))
    bottleneck = UpsampleBlock(
        in_channels=128, 
        out_channels=64,
        ratio=4, 
        dropout_rate=0.01)

    pred2 = bottleneck(x)
    print(pred2.shape)
