import torch
import numpy as np
import cv2
import os
from PIL import Image


class Compose(object):
    """
    Combine several transformation in a serial manner
    """
    def __init__(self, transform=None):
        self.transforms = transform

    def __call__(self, imgs, annos):
        if not self.transforms:
            return imgs, annos
        for m in self.transforms:
            imgs, annos = m(imgs, annos)
        return imgs, annos


class AddAxis(object):
    def __call__(self, imgs, annos):
        for idx, anno in enumerate(annos):
            annos[idx] = anno[:,:,np.newaxis]
        return imgs, annos


class Transpose(object):
    """
    transpose the image and mask
    """
    def __call__(self, imgs, annos):
        H, W, _ = imgs[0].shape
        if H < W:
            return imgs, annos
        else:
            timgs = [np.transpose(img, [1, 0, 2]) for img in imgs]
            tannos = [np.transpose(anno, [1, 0, 2]) for anno in annos]
            return timgs, tannos


# This will crop the original images, which only remains images containing needed features and then rescale
# version~0.0: update on Sep. 23
class Support_crop(object):
    def __init__(self, size):
        assert isinstance(size, (int, tuple, list))
        if isinstance(size, int):
            self.target_size = (size, size)
        else:
            self.target_size = size

    def __call__(self, imgs, annos):

        for img, ann in zip(imgs, annos):

            contours, hierarchy = cv2.findContours(ann, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            points = []
            for contour in contours:
                for [point] in contour:
                    points.append(point)
            x, y, w, h = cv2.boundingRect(np.array(points))

            # change w and h to the target ratio and pad
            # target_h / target_w
            if self.target_size[0] / self.target_size[1] > h / w:
                sh = int(w * self.target_size[0] / self.target_size[1])
                sw = w
            else:
                sh = h
                sw = int(h * self.target_size[1] / self.target_size[0])
            x, y = int(x - (sw - w) / 2), int(y - (sh - h) / 2)
            h, w = sh, sw
            x -= 20
            y -= 20
            h += 40
            w += 40

            # avoid the bound beyond limitation
            w = min(w, img.shape[1])
            h = min(h, img.shape[0])
            x = min(max(0, x), img.shape[1] - w)
            y = min(max(0, y), img.shape[0] - h)

            anno = ann.copy() * 255
            anno = cv2.cvtColor(anno, cv2.COLOR_GRAY2RGB)
            cv2.rectangle(anno, (x, y), (x + w, y + h), (0, 255, 2))

            # for contour in contours:
            #     for [point] in contour:
            #         anno = cv2.circle(anno, point, radius=1, color=(0, 0, 255), thickness=-1)
            # cv2.imshow("123", anno)
            # cv2.waitKey()
            img = img[y: y + h, x: x + w]
            ann = ann[y: y + h, x: x + w]
            cv2.imshow("1", img)
            cv2.imshow("2", ann * 255)
            cv2.waitKey()


class TrainTransform(object):
    def __init__(self, size):
        self.size = size
        self.transform = Compose([
            AddAxis(),
            Transpose(),
        ])

    def __call__(self, imgs, annos, support=False):
        pass


if __name__ == "__main__":
    from utility import *
    def load(path):
        images = []
        for i in range(5):
            img = np.array(Image.open(os.path.join(path, str(i) + '.png')))
            images.append(img)
        return images


    video_query_img = load(os.path.join(os.getcwd(), "tmp", 'video_query_img'))
    video_query_mask = load(os.path.join(os.getcwd(), "tmp", 'video_query_mask'))
    new_support_img = load(os.path.join(os.getcwd(), "tmp", 'new_support_img'))
    new_support_mask = load(os.path.join(os.getcwd(), "tmp", 'new_support_mask'))

    Support_crop([241, 425])(new_support_img, new_support_mask)

