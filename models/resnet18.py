import torch
from torch import nn
from torch.nn import functional as F 


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=strides, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        
        if in_channels != out_channels or strides != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut:
            x = self.shortcut(x)
        out += x

        return F.relu(out)


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    block_list = []

    for i in range(num_residuals):
        if i==0 and not first_block:
            block_list.append(ResidualBlock(in_channels, out_channels, strides=2))
        else:
            block_list.append(ResidualBlock(out_channels, out_channels))

    return block_list


def resnet18(num_classes=10, gray=False):
    num_channels = 3
    if gray == True:
        num_channels = 1

    model = nn.Sequential(nn.Conv2d(in_channels=num_channels, out_channels=64, 
                                    kernel_size=7, stride=2, padding=3), 
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                        *resnet_block(64, 64, 2, first_block=True),
                        *resnet_block(64, 128, 2),
                        *resnet_block(128, 256, 2),
                        *resnet_block(256, 512, 2),
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(),
                        nn.Linear(512, num_classes))
    
    return model


if __name__ == '__main__':
    net = resnet18()
    print(net)
    X = torch.randn(1, 3, 224, 224)
    print('\n Output shape:')
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, ':', X.shape)
