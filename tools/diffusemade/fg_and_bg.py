import argparse
import numpy as np
import cv2
import multiprocessing
import joblib
import os
import os.path as osp
import json

from compare_labels import get_file_list
from gen_label_and_prob import read_gray


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fg', type=str, required=True)
    parser.add_argument('--bg', type=str, required=True)
    parser.add_argument('--ct', type=str, default=None)
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--label-suffix', type=str, default='.png')
    parser.add_argument('--ignore', type=int, default=255)
    parser.add_argument('--n-jobs', type=int, default=None)
    parser.add_argument('--out', type=str, default=None)
    return parser.parse_args()


def main():
    n_jobs = multiprocessing.cpu_count() if args.n_jobs is None else args.n_jobs
    print(f"#jobs: {n_jobs}")

    file_list = get_file_list(args.fg, args.label_suffix, args.split)
    os.makedirs(args.out, exist_ok=True)
    out_ann_path = osp.join(args.out, "out_ann_dir")
    os.mkdir(out_ann_path)

    def process(i):
        name = file_list[i]
        fg_path = osp.join(args.fg, name + args.label_suffix)
        bg_path = osp.join(args.bg, name + args.label_suffix)
        out_path = osp.join(out_ann_path, name + args.label_suffix)
        fg = read_gray(fg_path)
        bg = read_gray(bg_path)
        ct = read_gray(osp.join(args.ct, name + args.label_suffix)) if args.ct is not None else np.zeros_like(fg)

        out = fg.copy()
        is_foreground = (fg != 0) & (fg != args.ignore)
        is_containing = (ct != 0) & (ct != args.ignore)
        is_background = (~is_foreground) & (bg != 0) & (bg != args.ignore) & (~is_containing)
        is_ignored = ~(is_foreground | is_background)
        out[is_ignored] = args.ignore
        out[is_background] = 0
        cv2.imwrite(out_path, out)

    meta_path = osp.join(args.out, "meta.json")
    with open(meta_path, 'w') as fp:
        json.dump(vars(args), fp)

    if n_jobs == 1:
        for i in range(len(file_list)):
            process(i)
    else:
        joblib.Parallel(n_jobs=n_jobs,
                        verbose=100,
                        pre_dispatch='all')(
            [joblib.delayed(process)(i) for i in range(len(file_list))]
        )


if __name__ == '__main__':
    args = parse_args()
    main()
