import torch
import torch.nn as nn
import math


def transformer(Q, K, V):
    """
    :param Q: [B, C, HW]
    :param K: [B, C, HW]
    :param V: [B, C, HW]
    :return:

    """
    C = Q.shape[1]
    P = torch.bmm(K.permute(0, 2, 1), Q)
    P = P / math.sqrt(C)
    P = torch.softmax(P, dim=1)
    M = torch.bmm(V, P)
    return M


class QueryKeyValue(nn.Module):
    def __init__(self, in_dim, key_dim, value_dim):
        super(QueryKeyValue, self).__init__()
        self.query = nn.Conv2d(in_dim, key_dim, kernel_size=3, padding=1, stride=1)
        self.key = nn.Conv2d(in_dim, key_dim, kernel_size=3, padding=1, stride=1)
        self.value = nn.Conv2d(in_dim, value_dim, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        return self.query(x), self.key(x), self.value(x)
