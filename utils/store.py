import os
import torch
import numpy as np
from main import args


def get_model_para_number(model):
    total = 0
    for para in model.parameters():
        total += torch.numel(para)
    return total


def save_checkpoint(model, epoch, loss, optimizer):
    if epoch % args.save_epoch != 0:
        return

    save_dir = os.path.join(args.snapshots_dir, "checkpoint")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    checkpoint_path = os.path.join(save_dir, 'checkpoint.pth.tar')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'loss': loss
    }, checkpoint_path)


def load_checkpoint(model, optimizer, checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(args.snapshots_dir, "checkpoint", 'checkpoint.pth.tar')
    assert os.path.exists(checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return epoch, loss


def save(model, model_path=None):
    if model_path is None:
        model_path = os.path.join(args.snapshots_dir, 'checkpoint', 'best_model.pth')
    model_dir = os.path.split(model_path)[0]
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_path)


def load(model, model_path=None):
    if model_path is None:
        model_path = os.path.join(args.snapshots_dir, 'checkpoint', 'best_model.pth')
    assert os.path.exists(model_path)
    model.load_state_dict(torch.load(model_path))


if __name__ == "__main__":
    from models.QueryKeyValue import QueryKeyValue
    qkv = QueryKeyValue(2, 2, 2)
    print(load_checkpoint(qkv, None))
