import argparse
import os
import os.path as osp

import matplotlib.pyplot as plt
from compare_labels import parse_refrain_info, get_file_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='voc')
    parser.add_argument('--data-info', type=str, required=True)
    parser.add_argument('--out', type=str, default=None, help='Path to out dir')
    return parser.parse_args()


def show_bars(xticks, y, save_path):
    assert len(xticks) == len(y)
    x = [i for i in range(len(y))]
    plt.bar(x, y)
    plt.xticks(x, xticks, rotation=60)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def main():
    out_path = None
    if args.out is not None:
        out_path = osp.join(args.out, "bar.png")

    cls_names, name2cls = parse_refrain_info(args.dataset, args.data_info)
    file_list = get_file_list(None, None, args.split)
    cls_bins = [0] * len(cls_names)
    for name in file_list:
        cls_bins[name2cls[int(name)]] += 1

    show_bars(cls_names[1:], cls_bins[1:], out_path)


if __name__ == '__main__':
    args = parse_args()
    main()
