from args import args
from PIL import Image
import random
import cv2
from pycocotools.ytvos import YTVOS
from torch.utils.data import Dataset
from args import args
import os
import numpy as np


class VosDataset(Dataset):
    def __init__(self, valid=False, transforms=None, test=False,
                 data_dir=args.data_dir, num_of_all_classes=args.num_of_all_classes, query_frame=args.query_frame,
                 sample_per_category=args.sample_per_category, support_frame=args.support_frame,
                 num_of_per_group=args.num_of_per_group, valid_idx=args.valid_idx):
        self.transforms = transforms
        self.img_dir = os.path.join(data_dir, 'Youtube-VOS', 'train', 'JPEGImages')
        self.ann_file = os.path.join(data_dir, 'Youtube-VOS', 'train', 'train.json')
        self.data_dir = data_dir
        self.num_of_all_classes = num_of_all_classes
        self.sample_per_category = sample_per_category
        self.num_of_per_group = num_of_per_group
        self.valid_idx = valid_idx
        self.support_frame = support_frame
        self.query_frame = query_frame
        self.test = test

        self.ytvos = YTVOS(self.ann_file)
        self.vid_infos = self.ytvos.vids
        self.load_annotations()
        print("Data set index:   {:d}.".format(self.valid_idx))

        if not valid and not test:
            self.category_list = [i + 1 for i in range(num_of_all_classes)
                                  if i % num_of_per_group != (valid_idx - 1)]
        else:
            self.category_list = [i + 1 for i in range(num_of_all_classes)
                                  if i % num_of_per_group == (valid_idx - 1)]

        self.category_vid_set = []
        for category in self.category_list:
            tmp_list = sorted(self.ytvos.getVidIds(catIds=category))
            self.category_vid_set.append(tmp_list)

        if not self.test:
            self.length = len(self.category_list) * sample_per_category
        else:
            self.length = 0
            self.test_support_category_vid_set = []
            self.test_support_frame = None
            self.test_support_mask = None

            for category_idx in range(len(self.category_list)):
                self.test_support_category_vid_set.append(random.sample(self.category_vid_set[category_idx],
                                                                        self.support_frame))
                for category_vid in self.test_support_category_vid_set[category_idx]:
                    self.category_vid_set[category_idx].remove(category_vid)
                self.length += len(self.category_vid_set[category_idx])

    def get_ground_truth_by_class(self, vid, category, frames_num=None):
        vid_info = self.vid_infos[vid]
        frames_list = vid_info['category_frames'][category].copy()  # which frame contains this category

        if frames_num is None:
            frames_num = len(frames_list)

        original_frames_len = len(frames_list)
        for i in range(frames_num - original_frames_len):
            frames_list.append(frames_list[original_frames_len - 1])
        choice_start_idx = random.randint(0, len(frames_list) - frames_num)
        choice_list = frames_list[choice_start_idx: choice_start_idx + frames_num]
        frames = [np.array(Image.open(os.path.join(self.img_dir, vid_info['file_names'][idx])))
                  for idx in choice_list]

        masks = []
        for idx in choice_list:
            objects_idx = vid_info['objects'][idx]
            mask = np.zeros((frames[0].shape[:2]), dtype=np.uint8)
            for object_idx in objects_idx:
                ann = self.ytvos.loadAnns(object_idx)[0]
                if ann['category_id'] != category:
                    continue
                mask += self.ytvos.annToMask(ann, idx)
            assert np.sum(mask) != 0
            mask = mask.clip(0, 1)
            masks.append(mask)

        return frames, masks

    def load_annotations(self):
        for vid, vid_info in self.vid_infos.items():
            vid_info['objects'] = [[] for i in range(vid_info['length'])]
            vid_info['category_frames'] = dict()
            annos = self.ytvos.vidToAnns[vid]
            for ann in annos:
                assert len(ann['segmentations']) == vid_info['length']
                for idx in range(vid_info['length']):
                    ann_seg = ann['segmentations'][idx]
                    if not ann_seg:
                        continue
                    vid_info['objects'][idx].append(ann['id'])
                    category = ann["category_id"]
                    if category in vid_info['category_frames']:
                        vid_info['category_frames'][category].append(idx)
                    else:
                        vid_info['category_frames'][category] = [idx]
            for key in vid_info['category_frames'].keys():
                vid_info['category_frames'][key] = list(set(vid_info['category_frames'][key]))

    def __len__(self):
        return self.length

    def get_train_item(self, item):
        category_idx = item // self.sample_per_category
        vid_set = self.category_vid_set[category_idx]

        query_vid = random.sample(vid_set, 1)
        support_vid = random.sample(vid_set, self.support_frame)

        query_frames, query_masks = self.get_ground_truth_by_class(query_vid[0],
                                                                   self.category_list[category_idx], self.query_frame)
        support_frames, support_masks = [], []
        for i in range(self.support_frame):
            frame, mask = self.get_ground_truth_by_class(support_vid[i], self.category_list[category_idx], 1)
            support_frames += frame
            support_masks += mask

        if self.transforms is not None:
            query_frames, query_masks = self.transforms(query_frames, query_masks)
            support_frames, support_masks = self.transforms(support_frames, support_masks, support=True)
        return query_frames, query_masks, support_frames, support_masks, self.category_list[category_idx]

    def get_test_item(self, item):
        id = item
        for category_idx in range(len(self.category_list)):
            if id >= len(self.category_vid_set[category_idx]):
                id -= len(self.category_vid_set[category_idx])
            else:
                if id == 0:
                    self.test_support_frame = []
                    self.test_support_mask = []
                    support_vid_set = self.test_support_category_vid_set[category_idx]
                    for support_vid in support_vid_set:
                        frame, mask = self.get_ground_truth_by_class(support_vid, self.category_list[category_idx], 1)
                        self.test_support_frame.append(frame[0])
                        self.test_support_mask.append(mask[0])
                    if self.transforms is not None:
                        self.test_support_frame, self.test_support_mask = \
                            self.transforms(self.test_support_frame, self.test_support_mask, support=True)

                vid = self.category_vid_set[category_idx][id]
                query_frames, query_masks = self.get_ground_truth_by_class(vid, self.category_list[category_idx])

                if self.transforms is not None:
                    query_frames, query_masks = self.transforms(query_frames, query_masks)

                return query_frames, query_masks, \
                       self.test_support_frame, self.test_support_mask, self.category_list[category_idx], \
                       os.path.split(self.vid_infos[vid]['file_names'][0])[0]

        return None

    def __getitem__(self, item):
        if not self.test:
            return self.get_train_item(item)
        else:
            return self.get_test_item(item)

    def get_category_list(self):
        return self.category_list


if __name__ == "__main__":
    from utility import show_img


    def save(imgs, path):
        if not os.path.exists(path):
            os.mkdir(path)
        for id, img in enumerate(imgs):
            Image.fromarray(img).save(os.path.join(path, str(id) + '.png'))


    from dataset.Transform import Transform

    transform = Transform(args.input_size)
    ytvos = VosDataset(test=True)
    for i in range(len(ytvos)):
        video_query_img, video_query_mask, new_support_img, new_support_mask, idx, name = ytvos[i]
        print("id:", i)

        # save(video_query_img, os.path.join(os.getcwd(), "tmp", "video_query_img"))
        # save(video_query_mask, os.path.join(os.getcwd(), "tmp", "video_query_mask"))
        # save(new_support_img, os.path.join(os.getcwd(), "tmp", "new_support_img"))
        # save(new_support_mask, os.path.join(os.getcwd(), "tmp", "new_support_mask"))
        #
        # break
