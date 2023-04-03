import argparse
import numpy as np
import os
import cv2
import json
import joblib
import multiprocessing

CLASSES = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor')

PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
           [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
           [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
           [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
           [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]


def main():
    n_jobs = multiprocessing.cpu_count() if args.n_jobs is None else args.n_jobs
    root_dir = args.root
    img_dir = os.path.join(root_dir, 'img_dir', 'train')
    ann_dir = args.ann
    assert os.path.isdir(ann_dir)
    cls_name2id = {}
    for id, name in enumerate(CLASSES):
        cls_name2id[name] = id
    palette = np.array(PALETTE)
    data_info_path = os.path.join(root_dir, 'data_info.json')
    with open(data_info_path, 'r') as fp:
        data_info = json.load(fp)
    show_dir = args.show_dir
    os.makedirs(show_dir, exist_ok=True)

    def process(i):
        di = data_info[i]
        cls_id = cls_name2id[di['concept']]
        pid = f"{di['img_index']:05}"
        label_path = os.path.join(ann_dir, pid + args.ann_suffix)
        label = cv2.imread(label_path, 0)
        is_ignore = label == 255
        label[is_ignore] = 0
        show_map = palette[label]
        show_map[is_ignore] = np.array([255, 255, 255])
        show_path = os.path.join(show_dir, pid + ".png")
        img_path = os.path.join(img_dir, pid + ".png")
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)[:, :, ::-1]  # BGR to RGB
        img = img * 0.5 + show_map * 0.5
        img = img[:, :, ::-1]  # RGB to BGR
        cv2.imwrite(show_path, img)

    joblib.Parallel(n_jobs=n_jobs,
                    verbose=100,
                    pre_dispatch='all')(
        [joblib.delayed(process)(i) for i in range(len(data_info))]
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/DiffuseMade3')
    parser.add_argument('--ann', type=str, default='data/DiffuseMade3/pseudo_masks_aug')
    parser.add_argument('--ann-suffix', type=str, default='.png')
    parser.add_argument('--show-dir', type=str, default='work_dirs/show/dm3_pseudo_masks_aug')
    parser.add_argument('--n-jobs', type=int, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main()
