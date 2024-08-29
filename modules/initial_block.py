import torch.nn as nn
import torch
class InitialBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InitialBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=out_channels-in_channels, kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.bn = nn.BatchNorm2d(out_channels-in_channels)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x_r = self.maxpool(x)
        x_l = self.bn(self.prelu(self.conv1(x)))
        x = torch.cat([x_r, x_l], 1)
        return x

if __name__ == '__main__':
    x = torch.randn((1, 3, 512, 512))
    initial = InitialBlock(3)
    pred = initial(x)