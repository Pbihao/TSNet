# @Author: Pbihao
# @Time  : 26/9/2021 11:03 AM

from easydict import EasyDict


class Measure_Log(EasyDict):
    def __init__(self, params=None, info=None):
        super(Measure_Log, self).__init__()
        self.params = params
        self.info = info
        self.total = 0
        for param in params:
            self[param] = 0

    def reset(self):
        for param in self.params:
            self[param] = 0

    def add(self, values, params=None, num=None):
        if params is None:
            params = self.params
        for idx, param in enumerate(params):
            self[param] += values[idx]
        self.total += num if num is not None else 1

    def get_average(self, params=None):
        result = EasyDict()
        if params is None:
            params = self.params
        for parm in params:
            result[parm] = self[parm] / self.total
        return result

    def print_average(self):
        if self.info is not None:
            print("~~>", self.info, ":")
        for param in self.params:
            print("    ", "{:<20}".format(param), ": %.4f" % (self[param] / self.total))
