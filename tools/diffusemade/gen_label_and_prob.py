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
from mmseg.core.evaluation.metrics import intersect_and_union, total_area_to_metrics

import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

from functools import partial

from compare_labels import print_results

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


# def print_results(results):
#     iou_list = results['IoU']
#     acc_list = results['Acc']
#     for cid, cname in enumerate(CLASSES):
#         print(f"{cid:5}, {cname:15}, {iou_list[cid] * 100:.2f}, {acc_list[cid] * 100:.2f}")
#
#     miou = np.nanmean(iou_list)
#     macc = np.nanmean(acc_list)
#     print(f"miou: {miou * 100:.2f}, macc: {macc * 100:.2f}")


def resize_ndarray(arr, **kwargs):
    t_arr = torch.from_numpy(arr)
    t_arr = t_arr.unsqueeze(0).unsqueeze(0)
    t_arr = F.interpolate(t_arr, **kwargs)
    return t_arr[0][0].cpu().numpy()


def act_power(x, p=1.0):
    return np.power(x, p)


def minmax_normalize(y):
    return (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-5)


def act_tanh(x, mid=0.5, sat=4):
    y = 1 / (1 + np.exp(-2 * sat * (x - mid)))
    return minmax_normalize(y)


def act_tanh2(x, mid=0.5, sat=4):
    up_range = 1.0 - mid
    lo_range = mid
    up_idx = x > mid
    lo_idx = x < mid
    y = x.copy()
    y[up_idx] = 1 / (1 + np.exp(- (sat / up_range) * (x[up_idx] - mid)))
    y[lo_idx] = 1 / (1 + np.exp(- (sat / lo_range) * (x[lo_idx] - mid)))
    return minmax_normalize(y)


def edge2positions(bin_edges):
    positions = []
    for i in range(len(bin_edges) - 1):
        positions.append((bin_edges[i] + bin_edges[i + 1]) / 2)
    return positions


def find_crests(hist, positions, window_size=0.1, is_trough=False, threshold=1000):
    crests = []
    crest_positions = []
    removed = [False] * len(hist)
    sorted_hist, sorted_positions = list(zip(*sorted(zip(hist, positions), key=lambda a: a[0], reverse=not is_trough)))
    for i in range(len(sorted_hist)):
        if len(crests) and (is_trough) ^ (sorted_hist[i] - threshold < 0):
            break
        if not removed[i]:
            crests.append(sorted_hist[i])
            crest_positions.append(sorted_positions[i])
            for j in range(i + 1, len(sorted_hist)):
                if abs(sorted_positions[j] - sorted_positions[i]) < window_size:
                    removed[j] = True
    return crests, crest_positions


def act_trough(x, win_size=0.1, th=1000, sat=4):
    hist, bin_edges = np.histogram(x, bins=100, range=(0, 1))
    positions = edge2positions(bin_edges)
    crest_positions = find_crests(hist, positions, win_size, is_trough=False, threshold=th)[1]
    trough_positions = find_crests(hist, positions, win_size, is_trough=True, threshold=th)[1]
    first_crest = min(crest_positions)
    mid = min(trough_positions)
    for p in sorted(trough_positions):
        if p > first_crest:
            mid = p
            break
    return act_tanh2(x, mid, sat)


def act_piece_wise(x, low, high):
    y = x.copy()
    y[y < low] = 0
    y[y > high] = 1
    return y


def act_softmax(x, mid=0.5, temperature=1.0):
    right_displace = mid - 0.5
    x2 = x - right_displace
    y = np.exp(x2 / temperature)  # element of y in [0, 1], so it is ok
    z = np.exp((1 - x2) / temperature)
    return y / (y + z)


def act_he(x, sat=4):
    x2 = cv2.equalizeHist((x * 255).astype(np.uint8)) / 255
    return act_tanh2(x2, 0.5, sat)


def read_gray(path):
    return cv2.imread(path, 0)


def read_npy(path):
    return np.load(path)


def main():
    n_jobs = multiprocessing.cpu_count() if args.n_jobs is None else args.n_jobs
    print(f"#jobs: {n_jobs}")

    pre_act = None
    if args.pre_act == 'pow':
        exp_name0 = f'{args.pre_act}-{args.pre_power}-{args.post}-{args.low}-{args.high}'
        pre_act = partial(act_power, p=args.pre_power)
    elif args.pre_act == 'tanh':
        exp_name0 = f'{args.pre_act}-{args.pre_mid}-{args.pre_sat}-{args.post}-{args.low}-{args.high}'
        pre_act = partial(act_tanh, mid=args.pre_mid, sat=args.pre_sat)
    elif args.pre_act == 'tanh2':
        exp_name0 = f'{args.pre_act}-{args.pre_mid}-{args.pre_sat}-{args.post}-{args.low}-{args.high}'
        pre_act = partial(act_tanh2, mid=args.pre_mid, sat=args.pre_sat)
    elif args.pre_act == 'piece':
        exp_name0 = f'{args.pre_act}-{args.pre_low}-{args.pre_high}-{args.post}-{args.low}-{args.high}'
        pre_act = partial(act_piece_wise, low=args.pre_low, high=args.pre_high)
    elif args.pre_act == 'softmax':
        exp_name0 = f'{args.pre_act}-{args.pre_mid}-{args.pre_temp}-{args.post}-{args.low}-{args.high}'
        pre_act = partial(act_softmax, mid=args.pre_mid, temperature=args.pre_temp)
    elif args.pre_act == 'trough':
        exp_name0 = f'{args.pre_act}-{args.pre_win}-{args.pre_th}-{args.pre_sat}-{args.post}-{args.low}-{args.high}'
        pre_act = partial(act_trough, win_size=args.pre_win, sat=args.pre_sat, th=args.pre_th)
    elif args.pre_act == 'he':
        exp_name0 = f'{args.pre_act}-{args.pre_sat}-{args.post}-{args.low}-{args.high}'
        pre_act = partial(act_he, sat=args.pre_sat)
    else:
        exp_name0 = f'{args.post}-{args.low}-{args.high}'
    # if args.test:
    #     exp_name = f'dm-test{args.root[-1]}-' + exp_name0
    # else:
    #     exp_name = f'dm{args.root[-1]}-' + exp_name0
    # if args.show_prob:
    #     exp_name += '-prob'

    root_dir = args.root
    data_info_dir = os.path.join(root_dir, 'data_info.json')
    img_dir = os.path.join(root_dir, args.img_dir)
    ann_dir = os.path.join(root_dir, args.ann_dir)
    out_ann_dir = os.path.join(root_dir, args.out_dir, 'out_ann_dir', exp_name0)
    show_dir = os.path.join(root_dir, args.out_dir, 'show', exp_name0)
    sta_dir = os.path.join(root_dir, args.out_dir, 'statistics', exp_name0)
    refer_dir = os.path.join(root_dir, args.refer)

    if args.ann_suffix == '.png':
        ann_reader = read_gray
    else:
        ann_reader = read_npy

    # refer_dir = args.refer
    # show_dir = None
    if args.show:
        os.makedirs(show_dir, exist_ok=True)
    if not args.eval_only:
        os.makedirs(out_ann_dir, exist_ok=True)

    os.makedirs(sta_dir, exist_ok=True)
    sta_name = 'deeplabv3'  # TODO: do not fix
    if args.refrain:
        sta_name += '*'
    sta_path = os.path.join(sta_dir, sta_name + '.txt')

    cls_name2id = {}
    for id, name in enumerate(CLASSES):
        cls_name2id[name] = id

    palette_bgr = np.array(PALETTE)[:, ::-1]
    mean_bgr = (104.008, 116.669, 122.675)

    if args.show_prob:
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

    data_info_path = os.path.join(root_dir, 'data_infos.json')
    with open(data_info_path, 'r') as fp:
        data_info = json.load(fp)

    num_classes = len(CLASSES)

    postprocess = None
    if args.post == 'dcrf':
        postprocess = DenseCRF(iter_max=10,
                               pos_xy_std=1,  # 1
                               pos_w=3,
                               bi_xy_std=67,  # 67
                               bi_rgb_std=3,  # 3
                               bi_w=4, )

    metrics = args.eval
    assert metrics is not None

    def process(i):
        di = data_info[i]
        cls_id = cls_name2id[di['concept']]
        refer_name = f"{di['img_index']:08}.png"
        ann_path = os.path.join(ann_dir, f"{di['img_index']:08}{args.ann_suffix}")
        refer_ann_path = os.path.join(refer_dir, refer_name)
        assert os.path.isfile(ann_path), f"No such file: {ann_path}"
        assert os.path.isfile(refer_ann_path), f"No such file: {refer_ann_path}"
        refer_ann = read_gray(refer_ann_path)
        if args.refrain:
            refer_ignored = ~((refer_ann == cls_id) | (refer_ann == 0))
            refer_ann[refer_ignored] = 255
        gray_ann = ann_reader(ann_path).astype(np.float32)
        gray_ann = resize_ndarray(gray_ann, size=refer_ann.shape, mode='bilinear')
        prob = gray_ann / 255.0
        # if abs(args.pre_power - 1.0) > 1e-7:
        if pre_act is not None:
            prob = pre_act(prob)
        if postprocess is not None:
            prob = np.stack([1 - prob, prob], axis=0)
            img_path = os.path.join(img_dir, f"{di['img_index']:08}{args.img_suffix}")
            img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32)
            img -= mean_bgr
            img = img.astype(np.uint8)
            prob = postprocess(img, prob)
            prob = prob[1]

        # if abs(args.power - 1.0) > 1e-7:
        # prob = np.power(prob, args.power)

        label = np.full(refer_ann.shape, 255, dtype=np.uint8)
        is_background = prob <= args.low
        is_foreground = prob >= args.high
        is_ignored = ~(is_background | is_foreground)
        label[is_background] = 0
        label[is_foreground] = cls_id
        ai, au, ap, al = intersect_and_union(label, refer_ann, num_classes, ignore_index=255)
        if not args.eval_only:
            out_label_path = os.path.join(out_ann_dir, f"{di['img_index']:08}.png")
            cv2.imwrite(out_label_path, label)
            if args.gen_prob:
                out_prob_path = os.path.join(out_ann_dir, f"{di['img_index']:08}_prob.npy")
                np.save(out_prob_path, prob)
        if show_dir is not None:
            is_ignore = label == 255
            label[is_ignore] = 0
            show_map = palette_bgr[label]
            show_map[is_ignore] = np.array([255, 255, 255])
            show_path = os.path.join(show_dir, f"{di['img_index']:08}.png")
            img_path = os.path.join(img_dir, f"{di['img_index']:08}.png")
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
    results = total_area_to_metrics(total_ai, total_au, total_ap, total_al, metrics=metrics)
    results['Area_i'] = total_ai.numpy()
    results['Area_u'] = total_au.numpy()
    results['Area_l1'] = total_ap.numpy()
    results['Area_l2'] = total_al.numpy()
    print_results(results, CLASSES, sta_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/DiffuseMade_test1')
    # after root
    parser.add_argument('--img-dir', type=str, default='img_dir/train')
    parser.add_argument('--ann-dir', type=str, default='ann_dir/train')
    parser.add_argument('--out-dir', type=str, default='.')
    parser.add_argument('--refer', type=str, default='deeplabv3-r50-d8_512x512_40k')

    parser.add_argument('--ann-suffix', type=str, default='.png', choices=['.png', '.npy'])
    parser.add_argument('--img-suffix', type=str, default='.png')

    parser.add_argument('--refrain', action='store_true')
    parser.add_argument('--low', type=float, default=0.05)
    parser.add_argument('--high', type=float, default=0.95)
    parser.add_argument('--post', type=str, choices=['no', 'dcrf'], default='dcrf')
    parser.add_argument('--eval-only', action='store_true')
    # parser.add_argument('--power', type=float, default=1.0)
    parser.add_argument('--pre-act', type=str,
                        choices=['pow', 'tanh', 'tanh2', 'piece', 'softmax', 'he', 'trough', 'no'],
                        default='pow')
    parser.add_argument('--pre-power', type=float, default=1.0)
    parser.add_argument('--pre-mid', type=float, default=0.5)
    parser.add_argument('--pre-sat', type=float, default=4.0)
    parser.add_argument('--pre-low', type=float, default=0.25)
    parser.add_argument('--pre-high', type=float, default=0.75)
    parser.add_argument('--pre-temp', type=float, default=1.0)
    parser.add_argument('--pre-win', type=float, default=0.2)
    parser.add_argument('--pre-th', type=float, default=5000)
    parser.add_argument('--gen-prob', action='store_true')
    # parser.add_argument('--test', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--show-prob', action='store_true')
    parser.add_argument('--min-a', type=float, default=0.4)
    parser.add_argument('--max-a', type=float, default=0.8)
    parser.add_argument('--n-jobs', type=int, default=None)
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
             ' for generic datasets, and "cityscapes" for Cityscapes')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main()
