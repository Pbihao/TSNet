# @Author: Pbihao
# @Time  : 26/9/2021 12:20 PM

from utils.davis_JF import db_eval_iou
from utils.davis_JF import db_eval_boundary
from args import args


def eval_boundary_iou(query_mask, pred, thresh=args.pred_thresh):
    boundary_sum = 0
    iou_sum = 0
    total = 0

    for batch in range(query_mask.shape[0]):
        for idx in range(query_mask.shape[1]):
            y = query_mask[batch][idx].detach().cpu().numpy() > thresh
            predict = pred[batch][idx].detach().cpu().numpy() > thresh

            boundary_sum += db_eval_boundary(predict, y)
            iou_sum += db_eval_iou(predict, y)
            total += 1

    return boundary_sum / total, iou_sum / total
