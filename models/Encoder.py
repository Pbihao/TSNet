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
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

    def forward(self, in_f, in_mask=None):
        f = in_f
        c1 = self.layer0(f)
        r2 = self.layer1(c1)
        r3 = self.layer2(r2)
        r4 = self.layer3(r3)

        if in_mask is not None:
            in_mask = F.interpolate(in_mask, r4.shape[2:], mode='bilinear', align_corners=True)
            r4 = r4 * in_mask

        return r4, r3, r2, c1, in_f


# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#
#         resnet = models.resnet50(pretrained=True)
#         self.conv1 = resnet.conv1
#         self.bn1 = resnet.bn1
#         self.relu = resnet.relu
#         self.maxpool = resnet.maxpool
#         self.res2 = resnet.layer1  # 1/4, 256
#         self.res3 = resnet.layer2  # 1/8, 512
#         self.res4 = resnet.layer3  # 1/8, 1024
#
#     ############################################################################################
#     # ~~~~~~~~~~~~~~~~need to be tested which method to merge the in_f and mask is best
#     def forward(self, in_f, in_mask=None):
#         x = self.conv1(in_f)  # 1/2, 64
#         x = self.bn1(x)
#         c1 = self.relu(x)
#         x = self.maxpool(c1)  # 1/4, 64
#         r2 = self.res2(x)  # 1/4, 256
#         r3 = self.res3(r2)  # 1/8, 512
#         r4 = self.res4(r3)  # 1/16, 1024
#
#         if in_mask is not None:
#             in_mask = F.interpolate(in_mask, r4.shape[2:], mode='bilinear', align_corners=True)
#             r4 = r4 * in_mask
#
#         return r4, r3, r2, c1, in_f


if __name__ == "__main__":
    from tmp.tmp import load_from_folder
    from utility import *
    query_img, query_mask, support_img, support_mask = load_from_folder()

    encoder = Encoder()
    q_r4, q_r3, q_r2, q_c1, q_f = encoder(query_img)
    s_r4, s_r3, s_r2, s_c1, s_f = encoder(support_img, support_mask)

    # support_path = os.path.join(os.getcwd(), 'tmp', 'support')
    # save_normal_img(support_img[3], support_path + '/query.png')
    # save_normal_img(support_mask[3], support_path + '/support.png')
    #
    # save_features(s_r4[3], support_path + '/r4')

    idx = 4
    query_path = os.path.join(os.getcwd(), 'tmp', 'query')
    save_normal_img(query_img[idx], query_path + '/query_' + str(idx) + '.png')
    save_normal_img(query_mask[idx], query_path + '/mask_' + str(idx) + '.png')
    save_features(q_r4[idx], query_path + '/r4_' + str(idx))

    print("finished")
