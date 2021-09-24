import os

import torch.nn as nn
import torch.nn.functional as F
from Encoder import ResBlock


class Refine(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.resFS = ResBlock(out_planes, out_planes)
        self.resMM = ResBlock(out_planes, out_planes)

    def forward(self, f, pm):  # upsample pm to f.size
        s = self.resFS(self.convFS(f))
        m = s + F.interpolate(pm, size=s.shape[2:], mode='bilinear', align_corners=True)
        m = self.resMM(m)
        return m


class Decoder(nn.Module):
    def __init__(self, in_planes, mdim):
        super(Decoder, self).__init__()
        self.convFM = nn.Conv2d(in_planes, mdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.resMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(512, mdim)
        self.RF2 = Refine(256, mdim)

        self.pred2 = nn.Conv2d(mdim, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)

    # ################################################################################################
    # ##################### differences between softmax or just use one dimension
    def forward(self, r4, r3, r2, f):
        m4 = self.resMM(self.convFM(r4))
        m3 = self.RF3(r3, m4)  # out: 1/8, 256
        m2 = self.RF2(r2, m3)  # out: 1/4, 256

        p2 = self.pred2(F.relu(m2))

        p = F.interpolate(p2, size=f.shape[2:], mode='bilinear', align_corners=True)
        return p


if __name__ == "__main__":
    from models.Encoder import Encoder
    from tmp.tmp import load_from_folder
    from utility import *
    query_img, query_mask, support_img, support_mask = load_from_folder()

    encoder = Encoder()
    s_r4, s_r3, s_r2, s_c1, s_f = encoder(support_img, support_mask)

    decoder = Decoder(in_planes=1024, mdim=256)
    preds = decoder(s_r4, s_r3, s_r2, s_f)

    pred_path = os.path.join(os.getcwd(), 'tmp', 'pred')
    for idx, s_img in enumerate(support_img):
        save_normal_img(s_img, pred_path + '/' + str(idx) + '.png')
    for idx, pred in enumerate(preds):
        save_feature(pred[0], pred_path + '/pred_' + str(idx) + '.png')

    print("finished")
