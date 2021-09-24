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

    def forward(self, query_img, support_img, support_mask):
        # x = self.encoder(query_img)
        return support_mask
        # return x
