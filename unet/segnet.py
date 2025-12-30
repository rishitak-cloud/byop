import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x
    
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(ConvBlock(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, padding=1))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x, ind = self.pool(x)
        return x, ind

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth, classification=False):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.layers = nn.ModuleList()
        for i in range(depth):
            if i == depth - 1 and classification:
                self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if i==depth-1:
                self.layers.append(ConvBlock(in_channels, out_channels))
            else:
                self.layers.append(ConvBlock(in_channels, in_channels))
    
    def forward(self, x, ind):
        x = self.unpool(x, ind)
        for layer in self.layers:
            x = layer(x)
        return x

class segnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.e1 = EncoderBlock(in_channels, 32, 2)
        self.e2 = EncoderBlock(32, 64, 2)
        self.e3 = EncoderBlock(64, 128, 3)
        self.e4 = EncoderBlock(128, 256, 3)
        self.e5 = EncoderBlock(256, 512, 3)
        self.d5 = DecoderBlock(512, 256, 3)
        self.d4 = DecoderBlock(256, 128, 3)
        self.d3 = DecoderBlock(128, 64, 3)
        self.d2 = DecoderBlock(64, 32, 2)
        self.d1 = DecoderBlock(32, out_channels, 2, classification=True)
    
    def forward(self, x):
        x1, ind1 = self.e1(x)
        x2, ind2 = self.e2(x1)
        x3, ind3 = self.e3(x2)
        x4, ind4 = self.e4(x3)
        x5, ind5 = self.e5(x4)
        a5 = self.d5(x5, ind1)
        a4 = self.d4(a5, ind2)
        a3 = self.d3(a4, ind3)
        a2 = self.d2(a3, ind4)
        a1 = self.d1(a2, ind5)
        return a1