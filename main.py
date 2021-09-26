import argparse
import os
from train import train
from args import args


if __name__ == '__main__':
    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)

    if args.train:
        if args.train_from_best_model:
            train(best_model=True)
        elif args.train_from_last_checkpoint:
            train(checkpoint=True)
        else:
            train()
