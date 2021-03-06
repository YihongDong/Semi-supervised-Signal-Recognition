#encoding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# SSRCNN
class SSRCNN(nn.Module):
    def __init__(self, num_classes=1000,num_channels=2):
        super(SSRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=[5, 1], stride=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=[2, 1], stride=[2, 1], return_indices=True)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=[3, 1], stride=1, bias=False)

        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=[3, 1], stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(0.5)

        self.maxUnpool = nn.MaxUnpool2d(kernel_size=[2, 1], stride=[2, 1])

        self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=[3, 1], stride=1, bias=False)

        self.debn3 = nn.BatchNorm2d(32)

        self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=[3, 1], stride=1, bias=False)

        self.debn2 = nn.BatchNorm2d(32)

        self.deconv1 = nn.ConvTranspose2d(32, 1, kernel_size=[5, 1], stride=1, bias=False)

        self.debn1 = nn.BatchNorm2d(1)

        self.fc1_u = nn.Linear(((((256//num_channels)-4)//2-2)//2-2)*16, 100)
        self.fc2 = nn.Linear(100, num_classes)

        self.defc1 = nn.Linear(100, ((((256//num_channels)-4)//2-2)//2-2)*16)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]* m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x, index = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x, index2 = self.maxpool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x, index3 = self.maxpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1_u(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def getSemantic(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x, index = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x, index2 = self.maxpool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x, index3 = self.maxpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1_u(x)

        return x

# construct SSRCNN
def getSSRCNN(**kwargs):
    model = SSRCNN(**kwargs)
    return model
