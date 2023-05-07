import os
import os.path as osp
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from gen_label_and_prob import read_gray, read_npy, resize_ndarray
from compare_labels import get_file_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann-dir', type=str, required=True)
    parser.add_argument('--ann-suffix', type=str, default='.png')
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--num-bins', type=int, default=100)
    return parser.parse_args()


def get_prob(ann_dir, name, ann_suffix, ann_reader):
    ann_path = osp.join(ann_dir, f"{name}{ann_suffix}")
    gray_ann = ann_reader(ann_path).astype(np.float32)
    gray_ann = resize_ndarray(gray_ann, size=(512, 512), mode='bilinear')
    prob = gray_ann / 255.0
    return prob


def main():
    file_list = get_file_list(args.ann_dir, args.ann_suffix, args.split)

    if args.ann_suffix == '.png':
        ann_reader = read_gray
    else:
        ann_reader = read_npy

    mean_hist = np.zeros(args.num_bins)
    var_hist = np.zeros(args.num_bins)
    bin_edges = None

    for i, name in tqdm(enumerate(file_list), desc="calculate mean", total=len(file_list)):
        prob = get_prob(args.ann_dir, name, args.ann_suffix, ann_reader)
        hist, bin_edges = np.histogram(prob, bins=args.num_bins, range=(0, 1))
        mean_hist = (mean_hist * i + hist) / (i + 1)

    for i, name in tqdm(enumerate(file_list), desc="calculate var", total=len(file_list)):
        prob = get_prob(args.ann_dir, name, args.ann_suffix, ann_reader)
        hist, bin_edges = np.histogram(prob, bins=args.num_bins, range=(0, 1))
        var_hist = (var_hist * i + (hist - mean_hist) ** 2) / (i + 1)

    std_hist = np.sqrt(var_hist)

    plt.stairs(mean_hist, bin_edges)
    plt.title("Mean")
    plt.show()

    plt.stairs(std_hist, bin_edges)
    plt.title("Std")
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    main()
