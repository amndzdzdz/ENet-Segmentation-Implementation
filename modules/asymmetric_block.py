import torch
import torch.nn as nn
import torch.nn.functional as F
from initial_block import Initial

class AsymmetricBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ratio, seq_length, dropout_rate):
        super(AsymmetricBlock, self).__init__()
        self.dropout_rate = dropout_rate
        self.inter_channels = in_channels//ratio
        self.seq_length = seq_length

        self.bn1 = nn.BatchNorm2d(self.inter_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()
        self.regularizer = nn.Dropout2d(dropout_rate)

        #PROJECTION 
        self.projection = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, stride=1)

        #CONV
        self.conv2 = self._make_asymmetric_conv(self.seq_length)

        #EXPANSION
        self.expansion = nn.Conv2d(self.inter_channels, out_channels, 1, 1)

    def _make_asymmetric_conv(self, seq_length):
        asymmetric_conv = []

        for _ in range(seq_length):
            asymmetric_conv.extend([self.prelu, self.bn1, 
                                    nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=(5,1), stride=1, padding=1)])
            asymmetric_conv.extend([self.prelu, self.bn1, 
                                    nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=(1,5), stride=1, padding=1)])

        return nn.Sequential(*asymmetric_conv)

    def forward(self, x): 
        #main path
        identity = x

        #right path
        x_r = self.bn1(self.prelu(self.projection(x)))
        x_r = self.conv2(x_r)
        x_r = self.bn2(self.prelu(self.expansion(x_r)))
        x_r = self.regularizer(x_r)

        return x_r + identity


if __name__ == "__main__":
    x = torch.randn((1, 128, 64, 64))
    bottleneck = AsymmetricBlock(
        in_channels=128, 
        out_channels=128,
        ratio=4,
        seq_length=5,
        dropout_rate=0.01)

    pred2 = bottleneck(x)
    print(pred2.shape)