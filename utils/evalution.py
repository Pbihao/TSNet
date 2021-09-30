# @Author: Pbihao
# @Time  : 26/9/2021 12:20 PM

from utils.davis_JF import db_eval_iou
from utils.davis_JF import db_eval_boundary
from args import args
import numpy as np


def eval_boundary_iou(query_mask, pred, thresh=args.pred_thresh):
    """
    :param query_mask: [F, H, W]
    :param pred: [F, H, W]
    :param thresh: float
    :return:  boundary_sum, iou_sum
    """
    boundary_sum = 0
    iou_sum = 0

    for idx in range(query_mask.shape[0]):
        y = query_mask[idx] > thresh
        predict = pred[idx] > thresh

        boundary_sum += db_eval_boundary(predict, y)
        iou_sum += db_eval_iou(predict, y)

    return boundary_sum, iou_sum


def eval_cross(query_mask, pred, thresh=args.pred_thresh):
    """
    :param query_mask: [F, H, W]
    :param pred: [F, H, W]
    :param thresh: float
    :return: tp, tn, fp, fn
    """
    query_mask = query_mask > thresh
    pred = pred > thresh
    tp = np.logical_and(query_mask, pred).sum()
    tn = np.logical_and(np.logical_not(query_mask), np.logical_not(pred)).sum()
    fp = np.logical_and(np.logical_not(query_mask), pred).sum()
    fn = np.logical_and(query_mask, np.logical_not(pred)).sum()
    return tp, tn, fp, fn
