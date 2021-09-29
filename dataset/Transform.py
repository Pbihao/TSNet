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


class ToFloat(object):
    """
    convert value type to float
    """
    def __call__(self, imgs, annos):
        for idx, img in enumerate(imgs):
            imgs[idx] = img.astype(dtype=np.float32, copy=True)
        for idx, anno in enumerate(annos):
            annos[idx] = anno.astype(dtype=np.float32, copy=True)
        return imgs, annos


class Rescale(object):
    """
    rescale the size of image and masks
    """
    def __init__(self, target_size):
        assert isinstance(target_size, (int, tuple, list))
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            self.target_size = target_size

    def __call__(self, imgs, annos):
        h, w = imgs[0].shape[:2]
        new_height, new_width = self.target_size

        factor = min(new_height / h, new_width / w)
        height, width = int(factor * h), int(factor * w)
        pad_l = (new_width - width) // 2
        pad_t = (new_height - height) // 2

        for id, img in enumerate(imgs):
            canvas = np.zeros((new_height, new_width, 3), dtype=np.float32)
            rescaled_img = cv2.resize(img, (width, height))
            canvas[pad_t:pad_t+height, pad_l:pad_l+width, :] = rescaled_img
            imgs[id] = canvas

        for id, anno in enumerate(annos):
            canvas = np.zeros((new_height, new_width, 1), dtype=np.float32)
            rescaled_anno = cv2.resize(anno, (width, height), cv2.INTER_NEAREST)
            canvas[pad_t:pad_t + height, pad_l:pad_l + width, :] = rescaled_anno[:, :, np.newaxis]
            annos[id] = canvas

        return imgs, annos


class Normalize(object):

    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3]).astype(np.float32)
        self.std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3]).astype(np.float32)

    def __call__(self, imgs, annos):

        for id, img in enumerate(imgs):
            imgs[id] = (img / 255.0 - self.mean) / self.std

        return imgs, annos


class Stack(object):
    """
    stack adjacent frames into input tensors
    """
    def __call__(self, imgs, annos):
        num_img = len(imgs)
        num_anno = len(annos)
        assert num_img == num_anno
        img_stack = np.stack(imgs, axis=0)
        anno_stack = np.stack(annos, axis=0)
        return img_stack, anno_stack


class ToTensor(object):
    """
    convert to torch.Tensor
    """
    def __call__(self, imgs, annos):
        imgs = torch.from_numpy(imgs.copy())
        annos = torch.from_numpy(annos.astype(np.uint8, copy=True)).float()
        imgs = imgs.permute(0, 3, 1, 2).contiguous()
        annos = annos.permute(0, 3, 1, 2).contiguous()
        return imgs, annos


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
        dst_imgs, dst_annos = [], []
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

            img = img[y: y + h, x: x + w]
            ann = ann[y: y + h, x: x + w]
            # cv2.imshow("1", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            # cv2.imshow("2", ann * 255)
            # k = cv2.waitKey()
            # if k == 27:
            #     quit()
            dst_imgs.append(img)
            dst_annos.append(ann)
        return dst_imgs, dst_annos


class Transform(object):
    def __init__(self, size):
        self.support_crop = Support_crop(size)
        self.transform = Compose([
            AddAxis(),
            ToFloat(),
            Rescale(size),
            Normalize(),
            Stack(),
            ToTensor()
        ])

    def __call__(self, imgs, annos, support=False):
        if support:
            imgs, annos = self.support_crop(imgs, annos)
        return self.transform(imgs, annos)


# tensor:[b, c, h, w] all elements are normalized  --> return list [narray, ... ]
# tensor:[c, h, w] --> return narray
def unnormalize_tensor_to_img(tensors: torch.Tensor):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if len(tensors.shape) == 3:
        tensors = tensors.unsqueeze(dim=0)
    images = []
    for tensor in tensors:
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)

        img = tensor.clamp(0, 1).permute((1, 2, 0)).cpu().numpy()
        img *= 255
        img = img.astype(np.uint8)
        images.append(img)
    return images if len(images) > 1 else images[0]


if __name__ == "__main__":

    from dataset.VosDataset import VosDataset
    ytvos = VosDataset()
    transform = Transform((241, 425))
