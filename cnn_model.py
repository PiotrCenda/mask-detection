import torch.nn as nn


class mask_net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(mask_net, self).__init__()

        # input: 3 X 256 x 256
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(4)) # out: 64 x 64 x 64
        
        self.res1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(64),
                                  nn.Dropout(p=0.3),
                                  nn.ReLU(inplace=True)) # out: 64 x 64 x 64

        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(4)) # out: 64 x 16 x 16
        
        self.res2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.Dropout(p=0.3),
                                  nn.ReLU(inplace=True)) # out: 64 x 16 x 16

        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(inplace=True),
                                  nn.AdaptiveAvgPool2d(output_size=1)) # out: 128 x 4 x 4

        self.classifier = nn.Sequential(nn.Flatten(),  # out: 128
                                        nn.Dropout(p=0.5),
                                        nn.Linear(128, 32),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(32, num_classes),
                                        nn.Softmax(dim=1))  # out: 2

    def forward(self, inpt):
        out = self.conv1(inpt)
        out = self.res1(out)
        out = self.conv2(out)
        out = self.res2(out)
        out = self.conv3(out)
        out = self.classifier(out)
        return out