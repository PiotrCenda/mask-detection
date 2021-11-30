import torch.nn as nn


class mask_net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(mask_net, self).__init__()

        # input: 3 X 256 x 256
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(16),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(4)) # out: 16 x 64 x 64

        self.conv2 = nn.Sequential(nn.Conv2d(16, 128, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(4)) # out: 128 x 16 x 16

        self.res1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(inplace=True)) # out: 128 x 16 x 16

        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(4)) # out: 256 x 4 x 4
        
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(512),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(4)) # out: 512 x 1 x 1

        self.classifier = nn.Sequential(nn.Flatten(),  # out: 512
                                        nn.Linear(512, 128),
                                        nn.Linear(128, num_classes))  # out: 2

    def forward(self, inpt):
        out = self.conv1(inpt)
        out = self.conv2(out)
        out = self.res1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.classifier(out)
        return out