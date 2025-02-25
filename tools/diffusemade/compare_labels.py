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
from mmseg.datasets import ADE20KDataset, PascalVOCDataset


def reduce_zero_label(gt_semantic_seg):
    # avoid using underflow conversion
    gt_semantic_seg[gt_semantic_seg == 0] = 255
    gt_semantic_seg = gt_semantic_seg - 1
    gt_semantic_seg[gt_semantic_seg == 254] = 255
    return gt_semantic_seg


def print_results(results, cls_names, out_path=None, reduce_zero=False):
    aAcc = results.pop('aAcc')
    out_str = ""
    eval_keys = results.keys()
    out_str += f"{'ID':<5}{'Name':<15}"
    beg_cid = 1 if reduce_zero else 0
    for i, k in enumerate(eval_keys):
        # if i == len(eval_keys) - 1:
        out_str += f"{k:<10}"
        # else:
        #     out_str += f"{k:<10},"
    out_str += "\n"
    for cid, cname in enumerate(cls_names):
        out_str += f"{cid + beg_cid:<5}{cname:<15}"
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


def parse_refrain_info(dataset, refrain_info):
    cls_names = None
    if dataset == 'voc':
        cls_names = PascalVOCDataset.CLASSES
    elif dataset == 'ade':
        cls_names = ADE20KDataset.CLASSES
    else:
        raise NotImplementedError()
    name2cls = {}
    if refrain_info is not None:
        cls_name2id = {}
        for id, name in enumerate(cls_names):
            cls_name2id[name] = id
        data_info_path = refrain_info
        with open(data_info_path, 'r') as fp:
            data_info = json.load(fp)
        for di in data_info:
            name2cls[int(di['img_index'])] = cls_name2id[di['concept']]

    return cls_names, name2cls


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
    parser.add_argument('--dataset', type=str, default='voc', choices=['voc', 'ade'])
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
             ' for generic datasets, and "cityscapes" for Cityscapes')

    return parser.parse_args()


def get_file_list(l1, label_suffix, split):
    file_list = []
    if split is not None:
        with open(split, 'r') as fp:
            while True:
                a_line = fp.readline()
                if not a_line:
                    break
                file_list.append(a_line.strip())
    else:
        for fname in os.listdir(l1):
            if fname.endswith(label_suffix):
                file_list.append(osp.splitext(fname)[0])
    return file_list


def main():
    file_list = get_file_list(args.l1, args.label_suffix, args.split)

    cls_names, name2cls = parse_refrain_info(args.dataset, args.refrain_info)

    reduce_zero = False
    if args.dataset == 'ade':
        reduce_zero = True
        assert args.ignore == 255

    num_classes = len(cls_names)

    def process(i):
        name = file_list[i]
        l1_path = osp.join(args.l1, name + args.label_suffix)
        l2_path = osp.join(args.l2, name + args.label_suffix)
        l1 = cv2.imread(l1_path, 0)
        l2 = cv2.imread(l2_path, 0)
        if reduce_zero:
            l1 = reduce_zero_label(l1)
            l2 = reduce_zero_label(l2)
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
    print_results(results, cls_names, args.out, reduce_zero)


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main()
