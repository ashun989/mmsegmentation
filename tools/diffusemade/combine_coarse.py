import argparse
import numpy as np
import cv2
import multiprocessing
import joblib
import os
import os.path as osp

from compare_labels import get_file_list

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('l1', type=str)
    parser.add_argument('l2', type=str)
    parser.add_argument('--split', type=str, default=None)  
    parser.add_argument('--label-suffix', type=str, default='.png')
    parser.add_argument('--out-suffix', type=str, choices=['.npy', '.png'], default='.npy')
    parser.add_argument('--n-jobs', type=int, default=None)
    parser.add_argument('--out', type=str, default=None)
    return parser.parse_args()


def standardization(x):
    return (x - np.mean(x)) / np.var(x)

def scale_up(x):
    y = (x - np.min(x)) / (np.max(x) - np.min(x))
    return y * 255

def save_png(out_path, x):
    cv2.imwrite(out_path, x.astype(np.uint8))

def save_npy(out_path, x):
    np.save(out_path, x)

def main(): 
    n_jobs = multiprocessing.cpu_count() if args.n_jobs is None else args.n_jobs
    print(f"#jobs: {n_jobs}")

    file_list = get_file_list(args.l1, args.label_suffix, args.split)
    os.makedirs(args.out, exist_ok=True)

    if args.out_suffix == '.png':
        saver = save_png
    else:
        saver = save_npy

    def process(i):
        name = file_list[i]
        l1_path = osp.join(args.l1, name + args.label_suffix)
        l2_path = osp.join(args.l2, name + args.label_suffix)
        out_path = osp.join(args.out, name + args.out_suffix)
        l1 = cv2.imread(l1_path, 0).astype(np.float32)
        l2 = cv2.imread(l2_path, 0).astype(np.float32)
        l_out = 2 * l1 * l2 / (l1 + l2 + 1e-5)
        l_out = scale_up(l_out)
        saver(out_path, l_out)

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