import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.Encoder import Encoder
from models.Decoder import Decoder


class TSNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query_img, support_img, support_mask):
        return support_mask
