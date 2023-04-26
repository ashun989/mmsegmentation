import argparse
import os
import os.path as osp
import json
import cv2
import multiprocessing
import joblib
import numpy as np

from compare_labels import get_file_list, parse_refrain_info, reduce_zero_label
from gen_label_and_prob import read_gray
from clip_score import TopkClassIO, VOC_BACKGROUND_CATEGORY

from mmseg.datasets.voc import PascalVOCDataset
from mmseg.core.evaluation.metrics import intersect_and_union, total_area_to_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, default='path/to/images')
    parser.add_argument('--img-suffix', type=str, default='.png')
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--method', type=str, choices=['seg', 'cls'], default='cls')
    # parser.add_argument('--num-bad', type=int, default=None)

    parser.add_argument('--seg-pred', type=str, default=None,
                        help='Path to pred ann directory, active if method is seg')
    parser.add_argument('--seg-gt', type=str, default=None,
                        help='Path to gt ann directory, active if method is seg')
    parser.add_argument('--seg-suffix', type=str, default='.png')
    parser.add_argument('--seg-th', type=float, default=0.20, help='Active if method is seg, default is 0.20')
    parser.add_argument('--seg-num', type=int, default=1000, help='Truncate number')
    parser.add_argument('--seg-pnum', type=int, default=500, help='Truncate number per class')
    parser.add_argument('--seg-refrain', action='store_true')
    parser.add_argument('--seg-method', type=str, choices=['num', 'score', 'pnum'])
    parser.add_argument('--seg-ignore', type=int, default=255)
    parser.add_argument('--seg-key', type=str, default='IoU',
                        choices=['IoU', 'Fscore', 'Dice', 'Acc', 'Precision', 'Recall'])

    parser.add_argument('--cls-file', type=str, default=None,
                        help='Path to class label file, active if method is cls')
    parser.add_argument('--cls-topk', type=int, default=1,
                        help='Active if method is cls, default is 1')
    parser.add_argument('--cls-th', type=float, default=0.80, help='Class score threshold')
    parser.add_argument('--cls-pnum', type=int, default=500, help='Truncate number per class')
    parser.add_argument('--cls-method', type=str, choices=['topk', 'score', 'pnum'], default='topk')

    parser.add_argument('--dataset', type=str, default='voc', choices=['voc', 'ade'])
    parser.add_argument('--data-info', type=str, default=None)
    parser.add_argument('--out', type=str, default=None, help='Path to output directory')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--sort', action='store_true')
    parser.add_argument('--n-jobs', type=int, default=None)
    return parser.parse_args()


def seg_filter(file_list):
    assert osp.isdir(args.seg_pred), f"No such dir: {args.seg_pred}"
    assert osp.isdir(args.seg_gt), f"No such dir: {args.seg_gt}"

    cls_names, name2cls = parse_refrain_info(args.dataset, args.data_info)
    reduce_zero = False
    if cls_names == 'ade':
        assert args.seg_ignore == 255
        reduce_zero = True
    num_classes = len(cls_names)

    def process(i):
        name = file_list[i]
        pred_path = osp.join(args.seg_pred, name + args.seg_suffix)
        gt_path = osp.join(args.seg_gt, name + args.seg_suffix)
        pred = read_gray(pred_path)
        gt = read_gray(gt_path)
        if reduce_zero:
            pred = reduce_zero_label(pred)
            gt = reduce_zero_label(gt)

        cls_id = name2cls[int(name)]
        if args.seg_refrain:
            gt_ignored = ~((gt == cls_id) | (gt == 0))
            gt[gt_ignored] = args.seg_ignore
        ai, au, ap, al = intersect_and_union(pred, gt, num_classes=num_classes, ignore_index=args.seg_ignore)
        results = total_area_to_metrics(ai, au, ap, al, metrics=['mIoU', 'mDice', 'mFscore'])
        return (name, cls_id, np.nanmean(results[args.seg_key]))

    n_jobs = multiprocessing.cpu_count() if args.n_jobs is None else args.n_jobs
    print(f"#jobs: {n_jobs}")

    if n_jobs > 1:
        results = joblib.Parallel(n_jobs=n_jobs,
                                  verbose=100,
                                  pre_dispatch='all')(
            [joblib.delayed(process)(i) for i in range(len(file_list))]
        )
    else:
        results = [process(i) for i in range(len(file_list))]


    if args.seg_method == 'num':
        results = sorted(results, key=lambda a: a[2])
        results = results[:args.seg_num]
    elif args.seg_method == 'score':
        results = sorted(results, key=lambda a: a[2])
        num_bad = 0
        while num_bad < len(results) and results[num_bad][2] <= args.seg_th:
            num_bad += 1
        results = results[:num_bad]
    else:  # pnum
        cls_bin = [0] * len(cls_names)
        for i, r in enumerate(results):
            cls_id = r[1]
            cls_bin[cls_id] += 1
        results = sorted(results, key=lambda a: (a[1], a[2]))
        remained = np.zeros(len(results), dtype=np.uint8)
        beg = 0
        for cls_num in cls_bin:
            remained[beg:beg + cls_num][:args.seg_pnum] = 1
            beg += cls_num
        results = [r for i, r in enumerate(results) if remained[i] == 1]

    if args.verbose:
        for i, r in enumerate(results):
            print(f"{i:08}: {r}")

    names = list(zip(*results))[0]
    return names


def cls_filter(file_list):
    assert osp.isfile(args.cls_file), f"No such file: {args.cls_file}"
    assert args.cls_topk <= 5
    topk_io = TopkClassIO(args.cls_file)
    results = topk_io.read_all()
    cls_names, name2cls = parse_refrain_info(args.dataset, args.data_info)
    all_clsnames = None
    if args.dataset == 'voc':
        all_clsnames = list(PascalVOCDataset.CLASSES) + VOC_BACKGROUND_CATEGORY
    rtn = []
    if args.cls_method == 'topk':
        for name, result in results.items():
            cls_id = name2cls[int(name)]
            labels = result['labels'][:args.cls_topk]
            probs = result['probs'][:args.cls_topk]
            if cls_id not in labels:
                rtn.append(name)
                if args.verbose:
                    print(f"{name}: ", end='')
                    for l, p in zip(labels, probs):
                        l_name = all_clsnames[l] if all_clsnames is not None else l
                        print(f"{l_name}={p}, ", end='')
                    print()
    elif args.cls_method == 'score':
        for name, result in results.items():
            cls_id = name2cls[int(name)]
            labels = result['labels']
            probs = result['probs']
            try:
                lidx = labels.index(cls_id)
                if probs[lidx] <= args.cls_th:
                    rtn.append((name, probs[lidx]))
            except ValueError:
                rtn.append((name, 0.0))
        rtn = sorted(rtn, key=lambda r: r[1])
        if args.verbose:
            for name, prob in rtn:
                print(f"{name}: {prob}")
        rtn = list(zip(*rtn))[0]
    else:  # pnum
        cls_bin = [0] * len(cls_names)
        for name, result in results.items():
            cls_id = name2cls[int(name)]
            cls_bin[cls_id] += 1
            labels = result['labels']
            probs = result['probs']
            try:
                lidx = labels.index(cls_id)
                rtn.append((name, cls_id, probs[lidx]))
            except ValueError:
                rtn.append((name, cls_id, 0.0))
        rtn = sorted(rtn, key=lambda r: (r[1], r[2]))
        remained = np.zeros(len(rtn), dtype=np.uint8)
        beg = 0
        for cls_num in cls_bin:
            remained[beg:beg + cls_num][:args.cls_pnum] = 1
            beg += cls_num
        rtn = [r for i, r in enumerate(rtn) if remained[i] == 1]
        if args.verbose:
            for name, cls_id, prob in rtn:
                print(f"{name}, {cls_id}, {prob}")
        rtn = list(zip(*rtn))[0]

    return rtn


def list_substract(l1, l2):
    return list(set(l1) - set(l2))


def write_file_list(file_list, path, sort=True):
    if sort:
        file_list = sorted(file_list)
    with open(path, 'w') as fp:
        for f in file_list:
            fp.write(f"{f}\n")


def main():
    file_list = get_file_list(args.img, args.img_suffix, args.split)

    os.makedirs(args.out, exist_ok=True)
    meta_path = osp.join(args.out, "meta.json")
    with open(meta_path, 'w') as fp:
        json.dump(vars(args), fp)

    if args.method == 'seg':
        bad_list = seg_filter(file_list)
    else:
        bad_list = cls_filter(file_list)

    good_list = list_substract(file_list, bad_list)

    bad_path = osp.join(args.out, "bad.txt")
    good_path = osp.join(args.out, "good.txt")
    write_file_list(bad_list, bad_path, sort=args.sort)
    write_file_list(good_list, good_path, sort=True)


if __name__ == '__main__':
    args = parse_args()
    main()
