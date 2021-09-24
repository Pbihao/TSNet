import torch.nn as nn


class QueryKeyValue(nn.Module):
    def __init__(self, in_dim, key_dim, value_dim):
        super(QueryKeyValue, self).__init__()
        self.query = nn.Conv2d(in_dim, key_dim, kernel_size=3, padding=1, stride=1)
        self.key = nn.Conv2d(in_dim, key_dim, kernel_size=3, padding=1, stride=1)
        self.value = nn.Conv2d(in_dim, key_dim, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        return self.query(x), self.key(x), self.value(x)