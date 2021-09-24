import os

import cv2
from matplotlib import pyplot as plt
import torch
import numpy as np
from dataset.Transform import unnormalize_tensor_to_img

# ********************************************************* SHOW ****************************


# input img: [:, :, :]    Tensor(w, h, c) or narray((w, h, c))
# img [[:, :, :], ]   list((w, h, c), )
def show_img(img):
    if type(img) == torch.Tensor:
        img = img.cpu().detach().numpy()

    if type(img) == list:
        for im in img:
            if type(im) == torch.Tensor:
                im = im.cpu().detach().numpy()
            plt.axis('off')
            plt.imshow(im)
            plt.show()
        return

    plt.axis('off')
    plt.imshow(img)
    plt.show()


# img: [b, c, w, h] or [c, w, h]
# all elements are normalized
def show_normal_img(img: torch.Tensor):
    img = unnormalize_tensor_to_img(img)
    show_img(img)


# feature: [c, w, h]
def show_feature(img: torch.Tensor):
    img = img.permute(1, 2, 0)
    show_img(img)


# ******************************************* SAVE  ***********************************


# img: torch.Tensor [c, w, h]
# all elements are normalized
def save_normal_img(img, path):
    img = unnormalize_tensor_to_img(img.clone())
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(path)


# features: [c, w, h]
# save all features under the folder with path of 'path'
def save_features(features, path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        ls = os.listdir(path)
        for name in ls:
            os.remove(os.path.join(path, name))
    for idx, feature in enumerate(features.clone()):
        feature = feature.cpu().detach().numpy()
        plt.clf()
        plt.imshow(feature)
        plt.axis('off')
        plt.colorbar()
        plt.savefig(os.path.join(path, str(idx) + '.png'))

        print("saved idx:", idx)


# feature: [w, h]
# path is not a folder but file
def save_feature(feature, path):
    feature = feature.cpu().detach().numpy()
    plt.clf()
    plt.imshow(feature)
    plt.axis('off')
    plt.colorbar()
    plt.savefig(path)
