import torch
import torch.nn as nn
from modules.asymmetric_block import AsymmetricBlock
from modules.downsample_block import DownsampleBlock
from modules.initial_block import InitialBlock
from modules.regular_block import RegularBlock
from modules.upsample_block import UpsampleBlock

class ENet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ENet, self).__init__()
        self.initial = InitialBlock(in_channels=in_channels, out_channels=16)

        self.bottleneck10 = DownsampleBlock(in_channels=16, out_channels=64, ratio=4, dropout_rate=0.01)
        self.bottleneck11 = RegularBlock(in_channels=64, out_channels=64, ratio=4, dropout_rate=0.01)
        self.bottleneck12 = RegularBlock(in_channels=64, out_channels=64, ratio=4, dropout_rate=0.01)
        self.bottleneck13 = RegularBlock(in_channels=64, out_channels=64, ratio=4, dropout_rate=0.01)
        self.bottleneck14 = RegularBlock(in_channels=64, out_channels=64, ratio=4, dropout_rate=0.01)

        self.bottleneck20 = DownsampleBlock(in_channels=64, out_channels=128, ratio=4, dropout_rate=0.01)
        self.bottleneck21 = RegularBlock(in_channels=128, out_channels=128, ratio=4, dropout_rate=0.01)
        self.bottleneck22 = RegularBlock(in_channels=128, out_channels=128, ratio=4, conv_type='dilated', dropout_rate=0.01, dilation=2)
        self.bottleneck23 = AsymmetricBlock(in_channels=128, out_channels=128, ratio=4, seq_length=5, dropout_rate=0.01)
        self.bottleneck24 = RegularBlock(in_channels=128, out_channels=128, ratio=4, dropout_rate=0.01, conv_type='dilated', dilation=4)
        self.bottleneck25 = RegularBlock(in_channels=128, out_channels=128, ratio=4, dropout_rate=0.01)
        self.bottleneck26 = RegularBlock(in_channels=128, out_channels=128, ratio=4, dropout_rate=0.01, conv_type='dilated', dilation=8)
        self.bottleneck27 = AsymmetricBlock(in_channels=128, out_channels=128, ratio=4, seq_length=5, dropout_rate=0.01)
        self.bottleneck28 = RegularBlock(in_channels=128, out_channels=128, ratio=4, dropout_rate=0.01, conv_type='dilated', dilation=16)
        
        self.bottleneck31 = RegularBlock(in_channels=128, out_channels=128, ratio=4, dropout_rate=0.01)
        self.bottleneck32 = RegularBlock(in_channels=128, out_channels=128, ratio=4, dropout_rate=0.01, conv_type='dilated', dilation=2)
        self.bottleneck33 = AsymmetricBlock(in_channels=128, out_channels=128, ratio=4, seq_length=5, dropout_rate=0.01)
        self.bottleneck34 = RegularBlock(in_channels=128, out_channels=128, ratio=4, dropout_rate=0.01, conv_type='dilated', dilation=4)
        self.bottleneck35 = RegularBlock(in_channels=128, out_channels=128, ratio=4, dropout_rate=0.01)
        self.bottleneck36 = RegularBlock(in_channels=128, out_channels=128, ratio=4, dropout_rate=0.01, conv_type='dilated', dilation=8)
        self.bottleneck37 = AsymmetricBlock(in_channels=128, out_channels=128, ratio=4, seq_length=5, dropout_rate=0.01)
        self.bottleneck38 = RegularBlock(in_channels=128, out_channels=128, ratio=4, dropout_rate=0.01, conv_type='dilated', dilation=16)

        self.bottleneck40 = UpsampleBlock(in_channels=128, out_channels=64, ratio=4, dropout_rate=0.01)
        self.bottleneck41 = RegularBlock(in_channels=64, out_channels=64, ratio=4, dropout_rate=0.01)
        self.bottleneck42 = RegularBlock(in_channels=64, out_channels=64, ratio=4, dropout_rate=0.01)
        
        self.bottleneck50 = UpsampleBlock(in_channels=64, out_channels=16, ratio=4, dropout_rate=0.01)
        self.bottleneck51 = RegularBlock(in_channels=16, out_channels=16, ratio=4, dropout_rate=0.01)

        self.linear = nn.Linear(in_features=1048576, out_features=out_channels)

    def forward(self, x):

        x = self.initial(x)

        x = self.bottleneck10(x)
        x = self.bottleneck11(x)
        x = self.bottleneck12(x)
        x = self.bottleneck13(x)
        x = self.bottleneck14(x)

        x = self.bottleneck20(x)
        x = self.bottleneck21(x)
        x = self.bottleneck22(x)
        x = self.bottleneck23(x)
        x = self.bottleneck24(x)
        x = self.bottleneck25(x)
        x = self.bottleneck26(x)
        x = self.bottleneck27(x)
        x = self.bottleneck28(x)

        x = self.bottleneck31(x)
        x = self.bottleneck32(x)
        x = self.bottleneck33(x)
        x = self.bottleneck34(x)
        x = self.bottleneck35(x)
        x = self.bottleneck36(x)
        x = self.bottleneck37(x)
        x = self.bottleneck38(x)

        x = self.bottleneck40(x)
        x = self.bottleneck41(x)
        x = self.bottleneck42(x)

        x = self.bottleneck50(x)
        x = self.bottleneck51(x)

        x = torch.flatten(x)

        x = self.linear(x)

        return x

if __name__ == "__main__":
    x = torch.randn((1, 3, 512, 512))

    model = ENet(
        in_channels=3, 
        out_channels=19)

    pred2 = model(x)
    print(pred2.shape)