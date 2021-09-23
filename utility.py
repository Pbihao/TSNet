import cv2
from matplotlib import pyplot as plt
import torch
# from libs.dataset.transform import ReverseToImage
import numpy as np

# input img: [:, :, :]    Tensor(w, h, c) or narray((w, h, c))
# img [[:, :, :], ]   list((w, h, c), )
def show_img(img):
    if type(img) == torch.Tensor:
        img = img.cpu().numpy()

    if type(img) == list:
        for im in img:
            if type(im) == torch.Tensor:
                im = im.cpu().numpy()
            plt.axis('off')
            plt.imshow(im)
            plt.show()
        return

    plt.axis('off')
    plt.imshow(img)
    plt.show()

# img: [:, :, :] torch.Tensor(c, w, h)
# def show_nor_img(img: torch.Tensor):
#     img = img.clone().detach()
#     img = ReverseToImage()(img)
#     img = img.cpu().numpy().transpose(1, 2, 0)
#     img = img * 255
#     img = img.astype(np.int)
#     show_img(img)