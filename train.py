import os
import torch
from utils.store import get_model_para_number
from models.TSNet import TSNet
from args import args
import sys
from utils.Logger import Logger
from utils.optimer import get_optimizer
from utils.store import load_checkpoint
from dataset.Transform import Transform
from dataset.VosDataset import VosDataset
from torch.utils.data import DataLoader
from utils.loss import cross_entropy_loss, mask_iou_loss
from utils.Measure_Log import Measure_Log
from utils.store import *
from utils.evalution import eval_boundary_iou
from tqdm import tqdm


def open_log_file(log_path=None):
    if log_path is None:
        log_path = os.path.join(os.getcwd(), 'snapshots', 'log.txt')
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


def train(open_log=True, checkpoint=False, pretrained_model=False):
    """
    :param pretrained_model: set True to train from pretrained model
    :param open_log:    set True to write all infos to log file
    :param checkpoint:  set True to start train from last checkpoint
    :return:
    """
    args.snapshots_dir = os.path.join(args.snapshots_dir, 'valid_idx_{:d}'.format(args.valid_idx))
    if open_log:
        open_log_file()

    print('==> Train Model: ', args.arch)
    model = TSNet()
    print('    Number of total params: %.2fM.' % (get_model_para_number(model) / 1000000))
    model = turn_on_cuda(model)
    optimizer = get_optimizer(model)
    start_epoch = 0
    best_mean_iou = 0
    if pretrained_model:
        load_model(model)
    elif checkpoint:
        start_epoch, start_loss = load_checkpoint(model, optimizer)
        print("\n==> Training from last checkpoint ...")
        print("     Epoch: %d" % start_epoch)
        print("     Loss: %f" % start_loss)

    print('\n==> Preparing dataset ... ')
    transform = Transform(args.input_size)
    train_dataset = VosDataset(transforms=transform)
    valid_dataset = VosDataset(valid=True, transforms=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print('\n==> Setting loss')
    criterion = lambda pred, target: [cross_entropy_loss(pred, target), mask_iou_loss(pred, target)]
    print('\n==> Start training ... ')
    for epoch in range(start_epoch, args.max_epoch):
        print('\n==> Training epoch {:d}'.format(epoch))

        model.train()
        loss_measure = Measure_Log(['total_loss', 'cross_entropy_loss', 'iou_loss'],
                                   "The loss of Epoch {:d}".format(epoch))
        for query_img, query_mask, support_img, support_mask, idx in tqdm(train_loader):
            query_img, query_mask, support_img, support_mask = turn_on_cuda(query_img), turn_on_cuda(query_mask), \
                                                               turn_on_cuda(support_img), turn_on_cuda(support_mask)
            pred_map = model(query_img, support_img, support_mask)
            pred_map = pred_map.squeeze(2)
            query_mask = query_mask.squeeze(2)

            ce_loss, iou_loss = criterion(pred_map, query_mask)
            loss = 5 * ce_loss + iou_loss
            loss_measure.add([loss.item(), ce_loss.item(), iou_loss.item()])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_measure.print_average()
        mean_loss = loss_measure.get_average(['total_loss']).total_loss

        eval_measure = Measure_Log(['boundary', 'iou'],
                                   "The scores of boundary and iou measures of Epoch {:d}".format(epoch))
        with torch.no_grad():
            model.eval()
            for query_img, query_mask, support_img, support_mask, idx in tqdm(valid_loader):
                query_img, query_mask, support_img, support_mask = turn_on_cuda(query_img), turn_on_cuda(query_mask), \
                                                                   turn_on_cuda(support_img), turn_on_cuda(support_mask)
                pred_map = model(query_img, support_img, support_mask)
                pred_map = pred_map.squeeze(2)
                query_mask = query_mask.squeeze(2)

                boundary, iou = eval_boundary_iou(query_mask, pred_map)
                eval_measure.add([boundary, iou])

        eval_measure.print_average()
        save_checkpoint(model, epoch, mean_loss, optimizer)
        mean_iou = eval_measure.get_average(['iou']).iou
        mean_boundary = eval_measure.get_average(['boundary']).boundary
        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            save_model(model, mean_boundary, mean_iou)
            print("    < Best model update at epoch {:d}. >".format(epoch))

    close_log_file()

if __name__ == "__main__":
    train(open_log=False)
