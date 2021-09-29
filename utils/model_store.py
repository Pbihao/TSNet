import os
import shutil

import torch
import numpy as np
from args import args


# deal with the conflict of store method between different versions if pytorch
def save_under_different_version(model, path, compatibility=True):
    version = torch.__version__
    version = int(version.split('.')[1])

    torch.save(model, path)
    if version >= 6 and compatibility:
        folder = os.path.split(path)[0]
        file_name = os.path.split(path)[1]
        idx = file_name.index('.')
        file_name, suffix = file_name[:idx], file_name[idx:]
        path = folder + "/" + file_name + "_less_than_1.6" + suffix
        torch.save(model, path, _use_new_zipfile_serialization=False)


def get_model_para_number(model):
    total = 0
    for para in model.parameters():
        total += torch.numel(para)
    return total


def save_checkpoint(model, epoch, loss, optimizer, checkpoint_path=None):
    if epoch % args.save_epoch != 0:
        return

    if checkpoint_path is None:
        checkpoint_path = os.path.join(args.snapshots_dir, 'checkpoint', 'checkpoint.pth.tar')
    checkpoint_dir = os.path.split(checkpoint_path)[0]
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    save_under_different_version({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'loss': loss
    }, checkpoint_path)

    print("    < Store checkpoint at epoch {:d}. >".format(epoch))


def load_checkpoint(model, optimizer, checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(args.snapshots_dir, 'checkpoint', 'checkpoint.pth.tar')
    assert os.path.exists(checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return epoch, loss


def save_model(model, boundary=None, iou=None, model_path=None):
    if model_path is None:
        model_path = os.path.join(args.snapshots_dir, 'checkpoint', 'best_model.pth')
    model_dir = os.path.split(model_path)[0]
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    save_under_different_version({
        'model_state_dict': model.state_dict(),
        'iou': iou,
        'boundary': boundary
    }, model_path)


def load_model(model, model_path=None):
    if model_path is None:
        model_path = os.path.join(args.snapshots_dir, 'checkpoint', 'best_model.pth')
    assert os.path.exists(model_path)
    model_dict = torch.load(model_path)
    model.load_state_dict(model_dict['model_state_dict'])
    return model_dict['boundary'], model_dict['iou']


if __name__ == "__main__":
    from models.QueryKeyValue import QueryKeyValue
    qkv = QueryKeyValue(2, 2, 2)
    save_model(qkv, 0, 0)
    print(load_model(qkv))
