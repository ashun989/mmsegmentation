import argparse
import numpy as np
import os
import cv2
import joblib
import multiprocessing
import matplotlib.pyplot as plt
from tqdm import tqdm

from compare_labels import get_file_list
from gen_label_and_prob import resize_ndarray, read_gray


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, default='data/DiffuseMade3/img_dir/train')
    parser.add_argument('--ann', type=str, default='data/DiffuseMade3/pseudo_masks_aug')
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--ann-suffix', type=str, default='.png')
    parser.add_argument('--img-suffix', type=str, default='.png')
    parser.add_argument('--show-dir', type=str, default='work_dirs/show/dm3_pseudo_masks_aug')
    parser.add_argument('--ignore', type=int, default=255)
    parser.add_argument('--n-jobs', type=int, default=None)
    return parser.parse_args()


def main():
    n_jobs = multiprocessing.cpu_count() if args.n_jobs is None else args.n_jobs
    img_dir = args.img
    ann_dir = args.ann
    assert os.path.isdir(img_dir)
    assert os.path.isdir(ann_dir)
    file_list = get_file_list(ann_dir, args.ann_suffix, args.split)

    show_dir = os.path.join(args.show_dir, "contour")
    os.makedirs(show_dir, exist_ok=True)

    def process(i):
        name = file_list[i]
        img_path = os.path.join(img_dir, name + args.img_suffix)
        ann_path = os.path.join(ann_dir, name + args.ann_suffix)
        show_path = os.path.join(show_dir, name + args.img_suffix)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)[:, :, ::-1]  # BGR to RGB
        gray_ann = read_gray(ann_path).astype(np.float32)
        gray_ann = resize_ndarray(gray_ann, size=img.shape[:2], mode='bilinear')
        prob = gray_ann / 255.0
        plt.figure()
        plt.contour(prob, levels=np.arange(0.1, 1.1, 0.1))
        plt.imshow(img)
        # plt.show()
        plt.savefig(show_path)
        plt.close()

    # if n_jobs > 1:
    #     joblib.Parallel(n_jobs=n_jobs,
    #                     verbose=100,
    #                     pre_dispatch='all')(
    #         [joblib.delayed(process)(i) for i in range(len(file_list))]
    #     )
    # else:
    for i in tqdm(range(len(file_list))):
        process(i)


if __name__ == '__main__':
    args = parse_args()
    main()
