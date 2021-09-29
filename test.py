# @Author: Pbihao
# @Time  : 28/9/2021 8:52 PM
import os
import torch
from utils.model_store import get_model_para_number
from models.TSNet import TSNet
from args import args
import sys
from utils.Logger import Logger
from utils.optimer import get_optimizer
from utils.model_store import load_checkpoint
from dataset.Transform import Transform
from dataset.VosDataset import VosDataset
from torch.utils.data import DataLoader
from utils.loss import cross_entropy_loss, mask_iou_loss
from utils.Measure_Log import Measure_Log
from utils.model_store import *
from utils.evalution import eval_boundary_iou
from tqdm import tqdm


def open_log_file(log_path=None):
    if log_path is None:
        log_path = os.path.join(args.snapshots_dir, 'test_log.txt')
    log_dir = os.path.split(log_path)[0]
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    sys.stdout = Logger(log_path)


def close_log_file():
    if hasattr(sys.stdout, 'is_logger'):
        del sys.stdout


def turn_on_cuda(x):
    if not hasattr(x, 'cuda'):
        return x
    if torch.cuda.is_available() and not args.turn_off_cuda:
        return x.cuda()
    return x


def test(open_log=True):
    """
    :param open_log:    set True to write all infos to log file
    """
    args.snapshots_dir = os.path.join(args.snapshots_dir, 'valid_idx_{:d}'.format(args.valid_idx))
    if open_log:
        open_log_file()

    print('==> Test Model: ', args.arch)
    model = TSNet()
    print('    Number of total params: %.2fM.' % (get_model_para_number(model) / 1000000))
    load_model(model)
    model = turn_on_cuda(model)

    print('\n==> Preparing dataset ... ')
    transform = Transform(args.input_size)
    test_dataset = VosDataset(test=True, transforms=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    print('\n==> Start Testing ... ')

    eval_measure = Measure_Log(['boundary', 'iou'],
                               "The scores of boundary and iou measures", print_step=True)
    with torch.no_grad():
        model.eval()
        for query_imgs, query_masks, support_img, support_mask, idx in tqdm(test_dataloader):
            query_imgs, query_masks, support_img, support_mask = turn_on_cuda(query_imgs), turn_on_cuda(query_masks), \
                                                               turn_on_cuda(support_img), turn_on_cuda(support_mask)
            for id in range(0, query_imgs.shape[1], args.query_frame):
                query_img = query_imgs[:, id: id + args.query_frame, :]
                query_mask = query_masks[:, id: id + args.query_frame, :]
                pred_map = model(query_img, support_img, support_mask)
                pred_map = pred_map.squeeze(2)
                query_mask = query_mask.squeeze(2)

                boundary, iou, num = eval_boundary_iou(query_mask, pred_map)
                eval_measure.add([boundary, iou], num=num)

    eval_measure.print_average()
    close_log_file()


if __name__ == "__main__":
    test()
