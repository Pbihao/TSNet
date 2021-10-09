import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim is None:
            outdim = indim
        if indim == outdim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/8, 1024

    ############################################################################################
    # ~~~~~~~~~~~~~~~~need to be tested which method to merge the in_f and mask is best
    def forward(self, in_f, in_mask=None):
        x = self.conv1(in_f)  # 1/2, 64
        x = self.bn1(x)
        c1 = self.relu(x)
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)  # 1/4, 256
        r3 = self.res3(r2)  # 1/8, 512
        r4 = self.res4(r3)  # 1/16, 1024

        if in_mask is not None:
            in_mask = F.interpolate(in_mask, r4.shape[2:], mode='bilinear', align_corners=True)
            r4 = r4 * in_mask

        return r4, r3, r2, c1, in_f

if __name__ == "__main__":
    from tmp.tmp import load_from_folder
    from utility import *
    query_img, query_mask, support_img, support_mask = load_from_folder()

    encoder = Encoder()
    q_r4, q_r3, q_r2, q_c1, q_f = encoder(query_img)
    s_r4, s_r3, s_r2, s_c1, s_f = encoder(support_img, support_mask)


    idx = 4
    query_path = os.path.join(os.getcwd(), 'tmp', 'query')
    save_normal_img(query_img[idx], query_path + '/query_' + str(idx) + '.png')
    save_normal_img(query_mask[idx], query_path + '/mask_' + str(idx) + '.png')
    save_features(q_r4[idx], query_path + '/r4_' + str(idx))

    print("finished")
