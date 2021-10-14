import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import models.resnet as models


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        resnet = models.resnet50(pretrained=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2,
                                    resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

    def forward(self, in_f, in_mask=None):
        f = in_f
        c0 = self.layer0(f)
        r1 = self.layer1(c0)
        r2 = self.layer2(r1)
        r3 = self.layer3(r2)
        r4 = self.layer4(r3)

        if in_mask is not None:
            in_mask = F.interpolate(in_mask, r4.shape[2:], mode='bilinear', align_corners=True)
            r4 = r4 * in_mask

        return r4, r3, r2, r1, in_f

