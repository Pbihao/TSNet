import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.Encoder import Encoder
from models.Decoder import Decoder
from args import args


def convert_to_input(query_img, support_img, support_mask):
    """
    :param [B, F, C, H, W]
    :return: [B * F, C, H, W]
    """
    batch = args.batch_size
    query_frame = args.query_frame
    support_frame = args.support_frame

    query_img = query_img.view((batch * query_frame, *query_img.shape[-3:]))
    support_img = support_img.view((batch * support_frame, *support_img.shape[-3:]))
    support_mask = support_mask.view((batch * support_frame, *support_mask.shape[-3:]))
    return query_img, support_img, support_mask


def convert_to_output(pred_mask):
    """
    :param [B * F, C, W, H]
    :return: [B, F, C, W, H]
    """
    batch = args.batch_size
    query_frame = args.query_frame

    pred_mask = pred_mask.view((batch, query_frame, *pred_mask.shape[-3:]))
    return pred_mask


class TSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.sigmoid = nn.Sigmoid()

    def forward(self, query_img, support_img, support_mask):
        query_img, support_img, support_mask = convert_to_input(query_img, support_img, support_mask)
        query_r4, query_r3, query_r2, query_c1, query_in_f = self.encoder(query_img)

        support_r4 = self.encoder(support_img, support_mask)

        r4 = query_r4 * support_r4

        mask = self.decoder(r4, query_r3, query_r2, query_in_f)
        mask = self.sigmoid(mask)
        mask = convert_to_output(mask)
        return mask


if __name__ == "__main__":
    a = torch.ones((1, 5, 3, 12, 21))
    convert_to_input(a, a, a)
