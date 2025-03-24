import numpy as np
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):

        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding)

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = torch.relu_(self.bn(self.conv(x)))
        return x


class Classifier(nn.Module):
    def __init__(self, L, resolution, in_channels, out_channels, num_class = 4):
        super().__init__()
        self.conv1 = ConvBlock(in_channels=in_channels,out_channels=out_channels,kernel_size=(3, 3), padding=(1,1))
        self.conv2 = ConvBlock(in_channels=16,out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = ConvBlock(in_channels=16,out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.pool = nn.AvgPool2d(kernel_size=(6,1), stride=(1,1))

        self.linear1 = nn.Linear(in_features=L*resolution*32,out_features= 64)
        self.linear2 = nn.Linear(in_features=64,out_features=num_class,)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = nn.Flatten()(x)
        x =self.linear1(x)
        x = self.linear2(x)


        return x