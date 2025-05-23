
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class PolygonUNetDownClassifier(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512], num_outputs=20):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downs = nn.ModuleList()
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.clf1 = nn.Conv2d(features[-1]*2, 256, 3, 1, 1, bias=False)
        self.clf2 = nn.Conv2d(256, 32, 3, 1, 1, bias=False)
        self.clf3 = nn.Conv2d(32, 6, 3, 1, 1, bias=False)

        self.classifier = nn.Sequential(
            nn.Linear(6*8*8, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_outputs)
        )

    def forward(self, x):
        for down in self.downs:
            x = down(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        x = self.clf1(x)
        x = self.clf2(x)
        x = self.clf3(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
