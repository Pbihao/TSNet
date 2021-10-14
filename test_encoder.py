# @Author: Pbihao
# @Time  : 14/10/2021 2:43 PM
import torch
from models.Encoder import *

with torch.no_grad():
    import sys
    from utils.Logger import Logger
    sys.stdout = Logger("/home/pbihao/PycharmProjects/TSNet/tmp/cos_log.txt")
    from dataset.Transform import TestTransform, unnormalize_tensor_to_img
    from dataset.VosDataset import VosDataset
    from args import args
    import torch.nn as nn
    import numpy as np
    from pprint import pprint
    import pickle

    from matplotlib import pyplot as plt

    transform = TestTransform(args.input_size)
    dataset = VosDataset(transforms=transform)
    encoder = Encoder()

    def get_one_image_and_mask(category, trans=True):
        """
        :param category:
        :param trans:
        :return:  [B, C, H, W]
        """
        if category >= len(dataset.category_list):
            return None
        vid = random.sample(dataset.category_vid_set[category], 1)[0]
        frames, masks = dataset.get_ground_truth_by_class(vid, dataset.category_list[category], 1)
        if trans:
            frames, masks = transform(frames, masks)
        return frames, masks


    def get_two_image_and_mask(category, trans=True):
        """
        :param category:
        :param trans:
        :return:  [B, C, H, W]
        """
        if category >= len(dataset.category_list):
            return None
        vid0, vid1 = random.sample(dataset.category_vid_set[category], 2)
        frames0, masks0 = dataset.get_ground_truth_by_class(vid0, dataset.category_list[category], 1)
        frames1, masks1 = dataset.get_ground_truth_by_class(vid1, dataset.category_list[category], 1)
        frames0.append(frames1[0])
        masks0.append(masks1[0])
        if trans:
            frames0, masks0 = transform(frames0, masks0)
        return frames0, masks0


    def get_feature_vector(cat):
        """
        sample in a category set and return it's feature_vector
        :return: [c]
        """
        frames, masks = get_one_image_and_mask(cat)
        r_4, _, _, _, _ = encoder(frames, masks)

        fr = frames[0]
        fr = unnormalize_tensor_to_img(fr)
        plt.imshow(fr)
        plt.show()

        r176 = r_4[:, 176, :]
        r176 = r176.view(*r176.shape[-2:])
        plt.imshow(r176)
        plt.show()

        r484 = r_4[:, 484, :]
        r484 = r484.view(*r484.shape[-2:])
        plt.imshow(r484)
        plt.show()

        avg = nn.AvgPool2d(r_4.shape[-2:], stride=1)

        r4 = avg(r_4)
        r4 = r4.permute((0, 2, 3, 1))
        r4 = r4.view(r4.shape[-1])
        return r4


    def test(cat=None):
        if cat is None:
            cat = [0, 0, 1]
        input_img = []
        input_mask = []

        frames, masks = get_two_image_and_mask(cat[0])
        input_img.append(frames[0])
        input_mask.append(masks[0])
        input_img.append(frames[1])
        input_mask.append(masks[1])
        frames, masks = get_one_image_and_mask(cat[2])
        input_img.append(frames[0])
        input_mask.append(masks[0])

        in_f = torch.stack(input_img, dim=0)
        in_m = torch.stack(input_mask, dim=0)
        r_4, _, _, _, _ = encoder(in_f, in_m)
        return r_4, input_img

    def calc_rel(r_4: torch.Tensor):
        avg = nn.AvgPool2d(r_4.shape[-2:], stride=1)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        r4 = avg(r_4)
        r4 = r4.permute((0, 2, 3, 1))
        r4 = r4.view(r4.shape[0], 1, r4.shape[-1])
        # print(r4.shape)
        ans = torch.zeros((len(r4), len(r4)))
        for i in range(len(r4)):
            for j in range(i + 1, len(r4)):
                a = r4[i]
                b = r4[j]
                ans[i][j] = ans[j][i] = cos(a, b)
        return ans
    #
    # res = {}
    # for i in range(10):
    #     res[i] = []
    # torch.set_printoptions(threshold=10000)
    # for i in range(1000):
    #     vecs = []
    #     for j in range(30):
    #         vec = get_feature_vector(j)
    #         vecs.append(vec)
    #     vecs = torch.stack(vecs, dim=0)
    #     var = torch.var(vecs, dim=0, unbiased=False)
    #
    #     cb = sorted(zip(var.tolist(), list(range(1024))))
    #
    #     cd = cb[:10] + cb[-10:]
    #
    #     print(cb)
    #     print("====>Feature: Max  ", torch.max(vecs), "Min", torch.min(vecs))
    #     print("====>Var: Max", torch.max(var), torch.min(var))
    #
    #     break

    cnt_Y = 0
    cnt_N = 0
    for i in range(1000):
        a, b = random.sample(list(range(30)), 2)

        r_4, _ = test([a, a, b])
        # print(r_4.shape)
        res = calc_rel(r_4)
        print(res)

        if torch.max(res) != res[0, 1]:
            cnt_N += 1
            print("N========>Epoch {:d}:({:d}, {:d})".format(i, a, b))
        else:
            cnt_Y += 1
            print("Y========>Epoch {:d}:({:d}, {:d})".format(i, a, b))

    print("cnt of Y:", cnt_Y)
    print("cnt of N:", cnt_N)

    # frames, masks = get_one_image_and_mask(0)
    # print(frames.shape, masks.shape)
    # frames, masks = unnormalize_tensor_to_img(frames), unnormalize_tensor_to_img(masks)
    # plt.imshow(frames)
    # plt.show()
    # plt.imshow(masks)
    # plt.show()
    # print(frames.shape, masks.shape)

