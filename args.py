# @Author: Pbihao
# @Time  : 26/9/2021 2:22 PM
import argparse
import os
import json


def get_arguments():
    parser = argparse.ArgumentParser(description="TSNet")
    parser.add_argument("--arch", type=str, default='TSNet')

    # parameters of train
    parser.add_argument("--max_iters", type=int, default=3000)
    parser.add_argument("--step_iter", type=int, default=100)
    parser.add_argument("--save_epoch", type=int, default=5)

    # set to swap models
    parser.add_argument("--train", action='store_true')  # test -> train
    parser.add_argument("--debug", action='store_true')  # run -> debug
    parser.add_argument("--turn_off_cuda", action='store_true')  # cuda on -> off
    parser.add_argument("--save_prediction_maps", action='store_true')  # save prediction
    parser.add_argument("--train_from_last_checkpoint", action='store_true')  # checkpoint
    parser.add_argument("--train_from_pretrained_model", action='store_true')  # best model

    # parameters of dataset
    parser.add_argument('--input_size', type=list, default=[241, 425])  # size of input images
    parser.add_argument('--valid_idx', type=int, default=1)  # class id used for valid [1, 2, 3, 4]
    parser.add_argument('--num_of_all_classes', type=int, default=40)  # num of all classes used
    parser.add_argument('--num_of_per_group', type=int, default=4)  # num of how many categories used for once train
    parser.add_argument('--sample_per_category', type=int, default=100)  # how may folders will be used for one category
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument("--max_epoch", type=int, default=101)

    # hyper-parameters
    parser.add_argument("--query_frame", type=int, default=5)
    parser.add_argument("--support_frame", type=int, default=5)
    parser.add_argument("--pred_thresh", type=float, default=0.5)

    # related path
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.getcwd(), "data"))
    parser.add_argument("--snapshots_dir", type=str, default=os.path.join(os.getcwd(), "snapshots"))

    return parser.parse_args()


args = get_arguments()
