import argparse
import os
import json

import torch
from tqdm import tqdm
import joblib
import multiprocessing
import numpy as np
import cv2
import torch.nn.functional as F
from mmseg.core.evaluation.metrics import intersect_and_union, total_area_to_metrics, mean_iou

import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

CLASSES = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor')

PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
           [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
           [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
           [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
           [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]


class DenseCRF(object):
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        C, H, W = probmap.shape

        U = utils.unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)

        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        )

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))

        return Q


def print_results(results):
    iou_list = results['IoU']
    acc_list = results['Acc']
    for cid, cname in enumerate(CLASSES):
        print(f"{cid:5}, {cname:15}, {iou_list[cid] * 100:.2f}, {acc_list[cid] * 100:.2f}")

    miou = np.nanmean(iou_list)
    macc = np.nanmean(acc_list)
    print(f"miou: {miou * 100:.2f}, macc: {macc * 100:.2f}")


def resize_ndarray(arr, **kwargs):
    t_arr = torch.from_numpy(arr)
    t_arr = t_arr.unsqueeze(0).unsqueeze(0)
    t_arr = F.interpolate(t_arr, **kwargs)
    return t_arr[0][0].cpu().numpy()


def main():
    n_jobs = multiprocessing.cpu_count() if args.n_jobs is None else args.n_jobs
    print(f"#jobs: {n_jobs}")

    exp_name = f'dm{args.root[-1]}-{args.post}-{args.power}-{args.low}-{args.high}'
    if args.show_prob:
        exp_name += '-prob'

    root_dir = args.root
    data_info_dir = os.path.join(root_dir, 'data_info.json')
    img_dir = os.path.join(root_dir, 'img_dir', 'train')
    ann_dir = os.path.join(root_dir, 'ann_dir', 'train')
    out_ann_dir = os.path.join(root_dir, f'pseudo-{exp_name}', 'train') if not args.eval_only else None
    refer_dir = os.path.join(root_dir, args.refer)
    show_dir = None
    if args.show:
        show_dir = os.path.join('work_dirs', 'show', exp_name)
        os.makedirs(show_dir, exist_ok=True)
    cls_name2id = {}
    for id, name in enumerate(CLASSES):
        cls_name2id[name] = id

    palette_bgr = np.array(PALETTE)[:, ::-1]
    mean_bgr = (104.008, 116.669, 122.675)

    assert 0 < args.min_a < args.max_a < 1.0
    theta_1 = (args.max_a - args.min_a) / (1.0 - args.high)
    beta_1 = args.max_a - theta_1
    theta_2 = (args.max_a - args.min_a) / (args.low)
    beta_2 = args.max_a - theta_2

    if args.show and args.show_prob:
        print(f"f(p) = {theta_1}p + {beta_1}: ({args.high}, 1.0) -> ({args.min_a}, {args.max_a})")
        print(f"f(p) = {theta_2}p + {beta_2}: (1.0 - {args.low}, 1.0) -> ({args.min_a}, {args.max_a})")

    if not args.eval_only:
        os.makedirs(out_ann_dir, exist_ok=True)

    data_info_path = os.path.join(root_dir, 'data_info.json')
    with open(data_info_path, 'r') as fp:
        data_info = json.load(fp)

    num_classes = len(CLASSES)
    total_ai = torch.zeros((num_classes,), dtype=torch.float64)
    total_au = torch.zeros((num_classes,), dtype=torch.float64)
    total_ap = torch.zeros((num_classes,), dtype=torch.float64)
    total_al = torch.zeros((num_classes,), dtype=torch.float64)

    postprocess = None
    if args.post == 'crf':
        raise NotImplementedError()
    elif args.post == 'dcrf':
        postprocess = DenseCRF(iter_max=10,
                               pos_xy_std=1,
                               pos_w=3,
                               bi_xy_std=67,
                               bi_rgb_std=3,
                               bi_w=4, )

    def process(i):
        di = data_info[i]
        cls_id = cls_name2id[di['concept']]
        filename = f"{di['img_index']:05}.png"
        ann_path = os.path.join(ann_dir, f"{di['img_index']:05}.png")
        refer_ann_path = os.path.join(refer_dir, filename)
        refer_ann = cv2.imread(refer_ann_path, 0)
        gray_ann = cv2.imread(ann_path, 0).astype(np.float32)
        gray_ann = resize_ndarray(gray_ann, size=refer_ann.shape, mode='bilinear')
        prob = gray_ann / 255.0
        if postprocess is not None:
            prob = np.stack([1 - prob, prob], axis=0)
            img_path = os.path.join(img_dir, f"{di['img_index']:05}.png")
            img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32)
            img -= mean_bgr
            img = img.astype(np.uint8)
            prob = postprocess(img, prob)
            prob = prob[1]

        if args.power > 1.0:
            prob = np.power(prob, args.power)

        label = np.full(refer_ann.shape, 255, dtype=np.uint8)
        is_background = prob <= args.low
        is_foreground = prob >= args.high
        is_ignored = ~(is_background | is_foreground)
        label[is_background] = 0
        label[is_foreground] = cls_id
        ai, au, ap, al = intersect_and_union(label, refer_ann, num_classes, ignore_index=255)
        if not args.eval_only:
            out_label_path = os.path.join(out_ann_dir, f"{di['img_index']:05}.png")
            out_prob_path = os.path.join(out_ann_dir, f"{di['img_index']:05}_prob.npy")
            cv2.imwrite(out_label_path, label)
            np.save(out_prob_path, prob)
        if show_dir is not None:
            is_ignore = label == 255
            label[is_ignore] = 0
            show_map = palette_bgr[label]
            show_map[is_ignore] = np.array([255, 255, 255])
            show_path = os.path.join(show_dir, f"{di['img_index']:05}.png")
            img_path = os.path.join(img_dir, f"{di['img_index']:05}.png")
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR
            if args.show_prob:
                show_lu = prob.copy()
                show_lu[is_ignore] = 0.5
                show_lu[is_foreground] = theta_1 * show_lu[is_foreground] + beta_1
                show_lu[is_background] = theta_2 * (1 - show_lu[is_background]) + beta_2
                show_lu = np.expand_dims(show_lu, axis=2)
                img = img * (1 - show_lu) + show_map * show_lu
            else:
                img = img * 0.5 + show_map * 0.5
            cv2.imwrite(show_path, img)
        return ai, au, ap, al

    if n_jobs > 1:
        areas = joblib.Parallel(n_jobs=n_jobs,
                                verbose=100,
                                pre_dispatch='all')(
            [joblib.delayed(process)(i) for i in range(len(data_info))]
        )
    else:
        areas = [process(i) for i in range(len(data_info))]

    ais, aus, aps, als = zip(*areas)
    total_ai = torch.sum(torch.stack(ais, dim=0), dim=0)
    total_au = torch.sum(torch.stack(aus, dim=0), dim=0)
    total_ap = torch.sum(torch.stack(aps, dim=0), dim=0)
    total_al = torch.sum(torch.stack(als, dim=0), dim=0)

    results = total_area_to_metrics(total_ai, total_au, total_ap, total_al, metrics=['mIoU'])
    # print(results)
    print_results(results)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/DiffuseMade2')
    # parser.add_argument('--out', type=str, default='pseudo_0')
    parser.add_argument('--refer', type=str, default='pseudo_masks_aug')
    parser.add_argument('--low', type=float, default=0.05)
    parser.add_argument('--high', type=float, default=0.95)
    parser.add_argument('--post', type=str, choices=['no', 'crf', 'dcrf'], default='dcrf')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--power', type=float, default=1.0)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--show-prob', action='store_true')
    parser.add_argument('--min-a', type=float, default=0.2)
    parser.add_argument('--max-a', type=float, default=0.7)
    parser.add_argument('--n-jobs', type=int, default=None)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main()
