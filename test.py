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
from utils.model_store import *
from utils.evalution import eval_boundary_iou
from tqdm import tqdm
from utils.Evaluation_Log import Evaluation_Log
import cv2


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


def save_predicts(preds_map, query_map, name, id):
    """
    :param preds_map:  normal [B, F, H, W]
    :param query_map:  normal [B, F, H, W]
    :param name: Str the name of folder
    :param id: int
    """
    preds_map = preds_map.squeeze(0)
    query_map = query_map.squeeze(0)
    preds_map = preds_map > args.pred_thresh
    preds_map = preds_map.type(torch.int).detach().cpu().numpy()
    query_map = query_map.type(torch.int).detach().cpu().numpy()
    for idx in range(preds_map.shape[0]):
        pred = preds_map[idx].astype(np.uint8) * 255
        query = query_map[idx].astype(np.uint8) * 255
        pred_dir = os.path.join(args.data_dir, 'Youtube-VOS', 'test', 'Predictions', name)
        query_dir = os.path.join(args.data_dir, 'Youtube-VOS', 'test', 'Masks', name)
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        if not os.path.exists(query_dir):
            os.makedirs(query_dir)
        pred_path = os.path.join(pred_dir, "{:05d}.png".format((id + idx) * 5))
        query_path = os.path.join(query_dir, "{:05d}.png".format((id + idx) * 5))
        cv2.imwrite(query_path, query)
        cv2.imwrite(pred_path, pred)


def test(open_log=True, save_prediction_maps=False):
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
    # in TEST the batch_size of loader must be 1
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    print('\n==> Start Testing ... ')

    evaluation = Evaluation_Log(test_dataset.get_category_list(), print_step=True)
    with torch.no_grad():
        model.eval()
        for query_imgs, query_masks, support_img, support_mask, idx, name in tqdm(test_dataloader):
            query_imgs, query_masks, support_img, support_mask = turn_on_cuda(query_imgs), turn_on_cuda(query_masks), \
                                                               turn_on_cuda(support_img), turn_on_cuda(support_mask)
            for id in range(0, query_imgs.shape[1], args.query_frame):
                query_img = query_imgs[:, id: id + args.query_frame, :]
                query_mask = query_masks[:, id: id + args.query_frame, :]
                pred_map = model(query_img, support_img, support_mask)
                pred_map = pred_map.squeeze(2)
                query_mask = query_mask.squeeze(2)

                if save_prediction_maps:
                    save_predicts(pred_map, query_mask, name[0], id)
                evaluation.add(idx, query_mask, pred_map)

    evaluation.print_average("The score of the whole test process")
    close_log_file()


if __name__ == "__main__":
    test()
