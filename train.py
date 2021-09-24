import os
import torch
from utils.store import get_model_para_number
from models.TSNet import TSNet
from main import args
import sys
from utils.Logger import Logger
from utils.optimer import get_optimizer
from utils.store import load_checkpoint
from dataset.Transform import Transform
from dataset.VosDataset import VosDataset
from torch.utils.data import DataLoader
from utils.loss import cross_entropy_loss, mask_iou_loss


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
        return
    if torch.cuda.is_available() and not args.turn_off_cuda:
        return x.cuda()
    return x

def train(open_log=True, checkpoint=False):
    if open_log:
        open_log_file()

    print('==> Train Model: ', args.arch)
    model = TSNet()
    print('    Number of total params: %.2fM.' % (get_model_para_number(model) / 1000000))
    model = turn_on_cuda(model)
    optimizer = get_optimizer(model)
    start_epoch = 0
    if checkpoint:
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
    criterion = lambda pred, target, bootstrap=1: [cross_entropy_loss(pred, target, bootstrap),
                                                   mask_iou_loss(pred, target)]
    print('\n==>Start training ... ')
    for epoch in range(start_epoch, args.max_epoch):
        print('\n==> Training epoch {:d}'.format(epoch))
        model.train()
        for query_img, query_mask, support_img, support_mask, idx in train_loader:
            query_img, query_mask, support_img, support_mask = turn_on_cuda(query_img), turn_on_cuda(query_mask), \
                                                               turn_on_cuda(support_img), turn_on_cuda(support_mask)
            pred_map = model(query_img, support_img, support_mask)

            pred_map = pred_map.squeeze(2)
            query_mask = query_mask.squeeze(2)

            ce_loss, iou_loss = criterion(pred_map, query_mask)
            loss = 5 * ce_loss + iou_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            for query_img, query_mask, support_img, support_mask, idx in valid_loader:
                query_img, query_mask, support_img, support_mask = turn_on_cuda(query_img), turn_on_cuda(query_mask), \
                                                                   turn_on_cuda(support_img), turn_on_cuda(support_mask)
                pred_map = model(query_img, support_img, support_mask)
                pred_map = pred_map.squeeze(2)
                query_mask = query_mask.squeeze(2)
                


if __name__ == "__main__":
    train(open_log=False)
