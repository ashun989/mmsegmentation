import argparse
import json
import os
import os.path as osp
import cv2
import multiprocessing
import joblib
import torch
import numpy as np
from mmseg.core.evaluation.metrics import intersect_and_union, total_area_to_metrics

VOC_CLASSES = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor')


def print_results(results, cls_names, out_path=None):
    aAcc = results.pop('aAcc')
    out_str = ""
    eval_keys = results.keys()
    out_str += f"{'ID':<5}{'Name':<15}"
    for i, k in enumerate(eval_keys):
        # if i == len(eval_keys) - 1:
        out_str += f"{k:<10}"
        # else:
        #     out_str += f"{k:<10},"
    out_str += "\n"
    for cid, cname in enumerate(cls_names):
        out_str += f"{cid:<5}{cname:<15}"
        for k in eval_keys:
            if k[:4] == 'Area':
                out_str += f"{results[k][cid] * 100:<10.2e}"
            else:
                out_str += f"{results[k][cid] * 100:<10.2f}"
        out_str += "\n"
    for k in eval_keys:
        mk = f"m{k}"
        out_str += f"{mk}={np.nanmean(results[k])}\n"
    out_str += f"aAcc={aAcc}\n"
    print(out_str)
    if out_path is not None:
        out_dir = os.sep.join(out_path.split(os.sep)[:-1])
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, 'w') as fp:
            fp.write(out_str)


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('img', type=str, help='Path to Images')
    parser.add_argument('l1', type=str, help='Path to Labels 1, like pred')
    parser.add_argument('l2', type=str, help='Path to Labels 2, like gt')
    parser.add_argument('--refrain-info', type=str, default=None, help='use l1 name to refrain l2 cls')
    parser.add_argument('--split', type=str, default=None, help='Split file')
    # parser.add_argument('--img-suffix', type=str, default='.jpg')
    parser.add_argument('--label-suffix', type=str, default='.png')
    parser.add_argument('--ignore', type=int, default=255)
    parser.add_argument('--n-jobs', type=int, default=None)
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='voc', choices=['voc'])
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
             ' for generic datasets, and "cityscapes" for Cityscapes')

    return parser.parse_args()


def main():
    file_list = []
    if args.split is not None:
        with open(args.split, 'r') as fp:
            while True:
                a_line = fp.readline()
                if not a_line:
                    break
                file_list.append(a_line.strip())
    else:
        for fname in os.listdir(args.l1):
            if fname.endswith(args.label_suffix):
                file_list.append(osp.splitext(fname)[0])

    cls_names = None
    if args.dataset == 'voc':
        cls_names = VOC_CLASSES
    num_classes = len(cls_names)

    name2cls = {}
    if args.refrain_info is not None:
        cls_name2id = {}
        for id, name in enumerate(cls_names):
            cls_name2id[name] = id
        data_info_path = args.refrain_info
        with open(data_info_path, 'r') as fp:
            data_info = json.load(fp)
        for di in data_info:
            name2cls[int(di['img_index'])] = cls_name2id[di['concept']]

    def process(i):
        name = file_list[i]
        l1_path = osp.join(args.l1, name + args.label_suffix)
        l2_path = osp.join(args.l2, name + args.label_suffix)
        l1 = cv2.imread(l1_path, 0)
        l2 = cv2.imread(l2_path, 0)
        if name2cls:
            cls_id = name2cls[int(name)]
            refer_ignored = ~((l2 == cls_id) | (l2 == 0))
            l2[refer_ignored] = args.ignore
        return intersect_and_union(l1, l2, num_classes=num_classes, ignore_index=args.ignore)

    n_jobs = multiprocessing.cpu_count() if args.n_jobs is None else args.n_jobs
    print(f"#jobs: {n_jobs}")

    metrics = args.eval
    assert metrics is not None
    # print(metrics)

    if n_jobs > 1:
        areas = joblib.Parallel(n_jobs=n_jobs,
                                verbose=100,
                                pre_dispatch='all')(
            [joblib.delayed(process)(i) for i in range(len(file_list))]
        )
    else:
        areas = [process(i) for i in range(len(file_list))]
    ais, aus, aps, als = zip(*areas)
    total_ai = torch.sum(torch.stack(ais, dim=0), dim=0)
    total_au = torch.sum(torch.stack(aus, dim=0), dim=0)
    total_ap = torch.sum(torch.stack(aps, dim=0), dim=0)
    total_al = torch.sum(torch.stack(als, dim=0), dim=0)
    results = total_area_to_metrics(total_ai, total_au, total_ap, total_al, metrics=metrics)
    results['Area_i'] = total_ai.numpy()
    results['Area_u'] = total_au.numpy()
    results['Area_l1'] = total_ap.numpy()
    results['Area_l2'] = total_al.numpy()
    print_results(results, cls_names, args.out)


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main()
