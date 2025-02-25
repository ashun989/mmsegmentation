# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from https://github.com/open-
mmlab/mmdetection/blob/master/tools/analysis_tools/analyze_logs.py."""
import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns


def plot_curve(log_dicts, args):
    if args.backend is not None:
        plt.switch_backend(args.backend)
    sns.set_style(args.style)
    # if legend is None, use {filename}_{key} as legend
    legend = args.legend
    if legend is None:
        legend = []
        for json_log in args.json_logs:
            for metric in args.keys:
                legend.append(f'{json_log}_{metric}')
    assert len(legend) == (len(args.json_logs) * len(args.keys))
    metrics = args.keys

    num_metrics = len(metrics)
    for i, log_dict in enumerate(log_dicts):
        iters = list(log_dict.keys())
        for j, metric in enumerate(metrics):
            print(f'plot curve of {args.json_logs[i]}, metric is {metric}')
            # plot_epochs = []
            plot_iters = []
            plot_values = []
            # In some log files exist lines of validation,
            # `mode` list is used to only collect iter number
            # of training line.
            for iter in iters:
                iter_logs = log_dict[iter]
                if metric not in iter_logs.keys():
                    continue
                # if epoch_logs['mode'][idx] == 'train':
                plot_iters.append(iter)
                plot_values.append(iter_logs[metric])
            ax = plt.gca()
            label = legend[i * num_metrics + j]
            # if metric in ['mIoU', 'mAcc', 'aAcc']:
            #     ax.set_xticks(plot_epochs)
            #     plt.xlabel('epoch')
            #     plt.plot(plot_epochs, plot_values, label=label, marker='o')
            # else:
            plt.xlabel('iter')
            plt.plot(plot_iters, plot_values, label=label, linewidth=0.5)
        plt.legend()
        if args.title is not None:
            plt.title(args.title)
    if args.out is None:
        plt.show()
    else:
        print(f'save curve to: {args.out}')
        plt.savefig(args.out)
        plt.cla()


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    parser.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser.add_argument(
        '--keys',
        type=str,
        nargs='+',
        default=['mIoU'],
        help='the metric that you want to plot')
    parser.add_argument('--title', type=str, help='title of figure')
    parser.add_argument(
        '--legend',
        type=str,
        nargs='+',
        default=None,
        help='legend of each plot')
    parser.add_argument(
        '--backend', type=str, default=None, help='backend of plt')
    parser.add_argument(
        '--style', type=str, default='dark', help='style of plt')
    parser.add_argument(
        '--mode', type=str, default='val', choices=['train', 'val'], help='log mode')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()
    return args


def load_json_logs(json_logs, mode):
    # load and convert json_logs to log_dict, key is epoch, value is a sub dict
    # keys of sub dict is different metrics
    # value of sub dict is a list of corresponding values of all iterations
    log_dicts = [dict() for _ in json_logs]
    for json_log, log_dict in zip(json_logs, log_dicts):
        with open(json_log, 'r') as log_file:
            last_iter = 0
            for line in log_file:
                log = json.loads(line.strip())
                if 'mode' not in log:
                    continue
                iter = log.pop('iter')
                line_mode = log.pop('mode')
                if line_mode != mode:
                    last_iter = iter
                    continue
                log.pop('epoch')
                if line_mode == 'train':
                    log_dict[iter] = dict()
                    for k, v in log.items():
                        log_dict[iter][k] = v
                    last_iter = iter
                else:
                    log_dict[last_iter] = dict()
                    for k, v in log.items():
                        log_dict[last_iter][k] = v

    return log_dicts


def main():
    args = parse_args()
    json_logs = args.json_logs
    for json_log in json_logs:
        assert json_log.endswith('.json')
    log_dicts = load_json_logs(json_logs, args.mode)
    plot_curve(log_dicts, args)


if __name__ == '__main__':
    main()
