# @Author: Pbihao
# @Time  : 30/9/2021 9:48 AM
import os
import pickle

import torch

from utils.evalution import eval_cross, eval_boundary_iou
from easydict import EasyDict


class Evaluation_Log(object):

    def __init__(self, category_list, params=None, print_step=False):
        super(Evaluation_Log, self).__init__()
        if params is None:
            params = ['intersection', 'union', 'f_score', 'j_score', 'num']
        self.category_list = category_list
        self.category_length = len(self.category_list)
        self.category_record = {}
        self.params = params
        self.print_step = print_step
        self.step = 0

        for category in self.category_list:
            self.category_record[category] = EasyDict()
            for param in self.params:
                self.category_record[category][param] = 0

    def add(self, categories, query_masks, preds):
        """
        :param categories: list or tensor
        :param query_masks: [B, F, H, W]
        :param preds: [B, F, H, W]
        """
        query_masks = query_masks.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        for batch in range(len(categories)):
            category = categories[batch]
            if not isinstance(category, int):
                category = category.item()
            query_mask = query_masks[batch]
            pred = preds[batch]
            tp, tn, fp, fn = eval_cross(query_mask, pred)
            boundary_sum, iou_sum = eval_boundary_iou(query_mask, pred)
            self.category_record[category].intersection += tp
            self.category_record[category].union += tp + fn + fp
            self.category_record[category].f_score += boundary_sum
            self.category_record[category].j_score += iou_sum
            self.category_record[category].num += query_mask.shape[0]

            self.step += 1
            if self.print_step and self.step % 100 == 0:
                self.print_average("The score at step {:d}".format(self.step))

    def get_iou(self, category):
        if category not in self.category_record or self.category_record[category].union == 0:
            return None
        return self.category_record[category].intersection / self.category_record[category].union

    def get_f_score(self, category):
        if category not in self.category_record or self.category_record[category].num == 0:
            return None
        return self.category_record[category].f_score / self.category_record[category].num

    def get_j_score(self, category):
        if category not in self.category_record or self.category_record[category].num == 0:
            return None
        return self.category_record[category].j_score / self.category_record[category].num

    def get_mean_iou(self):
        sum_iou = 0
        num = 0
        for category in self.category_record.keys():
            iou = self.get_iou(category)
            if iou is None:
                continue
            sum_iou += iou
            num += 1
        return sum_iou / max(num, 1)

    def get_mean_f_score(self):
        sum_f = 0
        num = 0
        for category in self.category_record.keys():
            F = self.get_f_score(category)
            if F is None:
                continue
            sum_f += F
            num += 1
        return sum_f / max(num, 1)

    def get_mean_j_score(self):
        sum_j = 0
        num = 0
        for category in self.category_record.keys():
            J = self.get_j_score(category)
            if J is None:
                continue
            sum_j += J
            num += 1
        return sum_j / max(num, 1)

    def print_average(self, info=None):
        if info is not None:
            print("~~>", info, ":")
        print("    ", "{:<20}".format("IOU"), ": %.4f" % (self.get_mean_iou()))
        print("    ", "{:<20}".format("Boundary_f"), ": %.4f" % (self.get_mean_f_score()))
        print("    ", "{:<20}".format("Iou_J"), ": %.4f" % (self.get_mean_j_score()))

    def save(self, path=None):
        assert path is not None
        if not os.path.exists(path):
            os.makedirs(path)
        res = {'IOU': self.get_mean_iou(),
               'Boundary_f': self.get_mean_f_score(),
               'Iou_J': self.get_mean_j_score(),
               'category': self.category_list}
        with open(os.path.join(path, "result.pkl"), 'wb') as f:
            pickle.dump(res, f)



if __name__ == "__main__":
    # [1, 5, 2, 2]
    def test():
        pred = torch.sigmoid_(torch.randn((5, 5, 2, 2)))
        mask = torch.randint(0, 2, (5, 5, 2, 2))
        class_list = [1, 2, 3]
        from tmp.test_evaluation import TreeEvaluation
        import numpy as np
        a = Evaluation_Log(class_list)
        b = TreeEvaluation(class_list)
        cl = [1, 2, 2, 1, 3]
        a.add(cl, mask, pred)
        b.update_evl(cl, mask, pred)
        print(a.get_mean_iou())
        print(np.mean(b.iou_list))

        print(a.get_mean_f_score())
        print(np.mean(b.f_score))

        print(a.get_mean_j_score())
        print(np.mean(b.j_score))
    test()



