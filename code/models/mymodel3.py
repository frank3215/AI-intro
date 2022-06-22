import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel3(nn.Module):
    def __init__(self):
        super(MyModel3, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, bias=False)       # output becomes 26x26
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 24, 3, bias=False)      # output becomes 24x24
        self.conv2_bn = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 32, 3, bias=False)      # output becomes 22x22
        self.conv3_bn = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 40, 3, bias=False)      # output becomes 20x20
        self.conv4_bn = nn.BatchNorm2d(40)
        self.conv5 = nn.Conv2d(40, 48, 3, bias=False)      # output becomes 18x18
        self.conv5_bn = nn.BatchNorm2d(48)
        self.conv6 = nn.Conv2d(48, 56, 3, bias=False)     # output becomes 16x16
        self.conv6_bn = nn.BatchNorm2d(56)
        self.conv7 = nn.Conv2d(56, 64, 3, bias=False)    # output becomes 14x14
        self.conv7_bn = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64, 72, 3, bias=False)    # output becomes 12x12
        self.conv8_bn = nn.BatchNorm2d(72)
        self.conv9 = nn.Conv2d(72, 80, 3, bias=False)    # output becomes 10x10
        self.conv9_bn = nn.BatchNorm2d(80)
        self.conv10 = nn.Conv2d(80, 88, 3, bias=False)   # output becomes 8x8
        self.conv10_bn = nn.BatchNorm2d(88)
        self.fc1 = nn.Linear(88*8*8, 10, bias=False)
        self.fc1_bn = nn.BatchNorm1d(10)
    def get_logits(self, x):
        x = (x - 0.5) * 2.0
        conv1 = F.relu(self.conv1_bn(self.conv1(x)))
        conv2 = F.relu(self.conv2_bn(self.conv2(conv1)))
        conv3 = F.relu(self.conv3_bn(self.conv3(conv2)))
        conv4 = F.relu(self.conv4_bn(self.conv4(conv3)))
        conv5 = F.relu(self.conv5_bn(self.conv5(conv4)))
        conv6 = F.relu(self.conv6_bn(self.conv6(conv5)))
        conv7 = F.relu(self.conv7_bn(self.conv7(conv6)))
        conv8 = F.relu(self.conv8_bn(self.conv8(conv7)))
        conv9 = F.relu(self.conv9_bn(self.conv9(conv8)))
        conv10 = F.relu(self.conv10_bn(self.conv10(conv9)))
        flat1 = torch.flatten(conv10.permute(0, 2, 3, 1), 1)
        logits = self.fc1_bn(self.fc1(flat1))
        return logits
    def forward(self, x):
        logits = self.get_logits(x)
        return F.log_softmax(logits, dim=1)
