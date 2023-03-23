import json
import os.path
import cv2
import numpy as np
from tqdm import tqdm

from collections import OrderedDict

CLASSES = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor')


# def cls():
#     cls_names = OrderedDict()
#     with open('data/DiffuseMade0/data_info.json', 'r') as fp:
#         data_info = json.load(fp)
#     print(len(data_info))
#     for di in data_info:
#         name = di['concept']
#         if name not in cls_names:
#             cls_names[name] = len(cls_names)
#     print(cls_names)
#     print(CLASSES - cls_names.keys())


def main():
    root_dir = 'data/DiffuseMade0'
    data_info_dir = os.path.join(root_dir, 'data_info.json')
    out_ann_dir = os.path.join(root_dir, 'out_ann_dir', 'train')
    ann_dir = os.path.join(root_dir, 'ann_dir', 'train')
    os.makedirs(out_ann_dir, exist_ok=True)

    cls_name2id = {}
    for id, name in enumerate(CLASSES):
        cls_name2id[name] = id

    print(cls_name2id)

    with open(data_info_dir, 'r') as fp:
        data_info = json.load(fp)
    for di in tqdm(data_info):
        cls_id = cls_name2id[di['concept']]
        ann_path = os.path.join(ann_dir, f"{di['img_index']:05}.png")
        out_ann_path = os.path.join(out_ann_dir, f"{di['img_index']:05}.npy")
        gray_ann = cv2.imread(ann_path, 0).astype(np.float32)
        # gray_ann /= 255
        gray_ann /= 255.1
        gray_ann += cls_id
        np.save(out_ann_path, gray_ann)


if __name__ == '__main__':
    main()
