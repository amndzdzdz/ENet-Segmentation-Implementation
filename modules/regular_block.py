import torch
import torch.nn as nn
import torch.nn.functional as F
from initial_block import Initial

class RegularBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ratio, conv_type, dropout_rate):
        super(RegularBlock, self).__init__()
        self.conv_type = conv_type
        self.dropout_rate = dropout_rate
        self.inter_channels = in_channels//ratio

        #PROJECTION 
        self.projection = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, stride=1)

        #CONV
        if self.conv_type == 'regular':
            self.conv2 = nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=3, stride=1, padding=1)
        elif self.conv_type == 'dilated':
            self.conv2 = nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=3, stride=1, padding=2, dilation=2)

        #EXPANSION
        self.expansion = nn.Conv2d(self.inter_channels, out_channels, 1, 1)

        self.bn1 = nn.BatchNorm2d(self.inter_channels)
        self.bn2 = nn.BatchNorm2d(self.inter_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()
        self.regularizer = nn.Dropout2d(dropout_rate)

    def forward(self, x): 
        #main path
        identity = x

        #right path
        x_r = self.bn1(self.prelu(self.projection(x)))
        x_r = self.bn2(self.prelu(self.conv2(x_r)))
        x_r = self.bn3(self.prelu(self.expansion(x_r)))
        x_r = self.regularizer(x_r)

        #pad main path
        channel_diff = x_r.shape[1] - identity.shape[1]
        padding = torch.zeros(identity.shape[0], channel_diff, identity.shape[2], identity.shape[3])
        identity = torch.cat([identity, padding], 1)

        return x_r + identity


if __name__ == "__main__":
    x = torch.randn((1, 3, 512, 512))
    initial = Initial(3, 16)
    pred = initial(x)
    bottleneck = RegularBlock(
        in_channels=16, 
        out_channels=64,
        ratio=4,  
        conv_type="regular", 
        dropout_rate=0.01)

    pred2 = bottleneck(pred)
    print(pred2.shape)