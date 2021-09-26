import argparse
import os
from train import train
from args import args


if __name__ == '__main__':
    from train import train
    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)

    if args.train:
        train()
