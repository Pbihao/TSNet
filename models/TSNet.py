import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.Encoder import Encoder
from models.Decoder import Decoder
from args import args
from models.QueryKeyValue import QueryKeyValue, transformer


def merge_batch_frame(*imgs):
    """
    :param [B, F, C, H, W]
    :return: [B * F, C, H, W]
    """
    result = []
    for img in imgs:
        batch = args.batch_size
        frame = img.shape[1]
        img = img.view(batch * frame, *img.shape[-3:])
        result.append(img)
    return result[0] if len(result) == 1 else result


def split_batch_frame(img):
    """
    :param [B * F, C, H, W]
    :return: [B, F, C, H, W]
    """
    batch = args.batch_size
    frame = img.shape[0] // batch
    img = img.view((batch, frame, *img.shape[-3:]))
    return img


def merge_FWH(img):
    """
    :param img: [B * F, C, H, W] ot [B, F, C, H, W]
    :return:  [B, C, F * H * W]
    """
    if len(img.shape) == 4:
        img = split_batch_frame(img)
    img = img.transpose(1, 2).contiguous()
    img = img.view(*img.shape[:2], -1)
    return img


def split_FWH(img, shape: tuple):
    """
    :param shape: [H, W]
    :param img: [B, C, F * H * W]
    :return: [B, F, C, H, W]
    """
    H, W = shape
    frame = img.shape[2] // H // W
    img = img.view(*img.shape[:2], frame, H, W)
    img = img.transpose(1, 2).contiguous()
    return img


class TSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_feature = 1024
        self.num_value = 512
        self.num_key = 128

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.sigmoid = nn.Sigmoid()
        self.support_qkv = QueryKeyValue(in_dim=self.num_feature, key_dim=self.num_key, value_dim=self.num_value)
        self.query_qkv = QueryKeyValue(in_dim=self.num_feature, key_dim=self.num_key, value_dim=self.num_value)
        self.conv_q = nn.Conv2d(self.num_feature, self.num_value, kernel_size=1, stride=1, padding=0)

        self.feature_shape = None

    def forward(self, query_img, support_img, support_mask):
        query_img, support_img, support_mask = merge_batch_frame(query_img, support_img, support_mask)
        query_r4, query_r3, query_r2, query_c1, query_in_f = self.encoder(query_img)
        support_r4, _, _, _, _ = self.encoder(support_img, support_mask)

        _, support_k, support_v = self.support_qkv(support_r4)
        query_q, query_k, query_v = self.query_qkv(query_r4)
        self.feature_shape = tuple(query_q.shape[-2:])

        support_k = merge_FWH(support_k)  # [B, Ck, F * H * W]
        support_v = merge_FWH(support_v)  # [B, Cv, F * H * W]

        query_q = split_batch_frame(query_q)  # [B, F, Cq, H, W]
        mid_frame_idx = query_q.shape[1] // 2
        mid_query_q = query_q[:, mid_frame_idx]  # [B, Cq, H, W]
        mid_query_q = mid_query_q.view(*mid_query_q.shape[:2], -1)  # [B, Cq, H * W]

        mid_v = transformer(mid_query_q, support_k, support_v)  # [B, Cv, H * W]

        query_k = split_batch_frame(query_k)  # [B, F, Ck, H, W]
        mid_query_k = query_k[:, mid_frame_idx]  # [B, Ck, H, W]
        mid_query_k = mid_query_k.view(*mid_query_k.shape[:2], -1)  # [B, Ck, H * W]
        query_q = merge_FWH(query_q)  # [B, Cq, F * H * W]

        V = transformer(query_q, mid_query_k, mid_v)  # [B, Cv, F*H*W]
        V = split_FWH(V, self.feature_shape)  # [B, F, Cv, H, W]

        query_r4 = self.conv_q(query_r4)  # [B * F, Cv, H, W]
        V = merge_batch_frame(V)  # [B * F, Cv, H, W]
        query_r4 = torch.cat([V, query_r4], dim=1)  # [B * F, C, H, W]

        mask = self.decoder(query_r4, query_r3, query_r2, query_in_f)
        mask = self.sigmoid(mask)
        mask = split_batch_frame(mask)
        return mask


if __name__ == "__main__":
    a = torch.ones((1, 4, 3, 241, 421))
    TSNet().forward(a, a, a[:, :, 1, :].unsqueeze(2))
