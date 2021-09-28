# @Author: Pbihao
# @Time  : 26/9/2021 11:03 AM

from easydict import EasyDict


class Measure_Log(EasyDict):
    def __init__(self, params=None, info=None, print_step=False):
        super(Measure_Log, self).__init__()
        self.print_step = print_step
        self.params = params
        self.info = info
        self.N = 0
        for param in params:
            self[param] = 0

    def reset(self):
        for param in self.params:
            self[param] = 0

    def add(self, values, params=None):
        if params is None:
            params = self.params
        for idx, param in enumerate(self.params):
            self[param] += values[idx]
        self.N += 1

    def get_average(self, params=None):
        result = EasyDict()
        if params is None:
            params = self.params
        for parm in params:
            result[parm] = self[parm] / self.N
        return result

    def print_average(self):
        if self.info is not None:
            print("~~~~:", self.info, ":")
        for param in self.params:
            print("    ", "{:<20}".format(param), ": %.4f" % (self[param] / self.N))





if __name__ == "__main__":
    log = Measure_Log(['celoss', 'sc', 'p'], "Epoch {:d}".format(2))
    log.add([1, 2, 3])
    log.add([2, 3, 4])
    log.print_average()

    print(log.celoss)
    print(log.p)
