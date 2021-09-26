import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.Encoder import Encoder
from models.Decoder import Decoder


class TSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, query_img, support_img, support_mask):
        r4, r3, r2, c1, in_f = self.encoder(support_img, support_mask)
        mask = self.decoder(r4, r3, r2, in_f)
        return mask
