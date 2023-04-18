import os
import os.path as osp
import argparse
from itertools import islice
import visdom

# compare_list = [
#     "cross_th0.05-top_k_percent0.25-temperature0.1",
#     "cross_th0.05-top_k_percent0.25-temperature0.3",
#     "cross_th0.05-top_k_percent0.25-temperature0.5",
#     "cross_th0.05-top_k_percent0.25-temperatureNone",
#     "cross_th0.05-top_k_percent0.5-temperature0.1",
#     "cross_th0.05-top_k_percent0.5-temperature0.3",
#     "cross_th0.05-top_k_percent0.5-temperature0.5",
#     "cross_th0.05-top_k_percent0.5-temperatureNone",
#     "cross_th0.05-top_k_percent0.75-temperature0.1",
#     "cross_th0.05-top_k_percent0.75-temperature0.3",
#     "cross_th0.05-top_k_percent0.75-temperature0.5",
#     "cross_th0.05-top_k_percent0.75-temperatureNone",
#     "cross_th0.15-top_k_percent0.25-temperature0.1",
#     "cross_th0.15-top_k_percent0.25-temperature0.3",
#     "cross_th0.15-top_k_percent0.25-temperature0.5",
#     "cross_th0.15-top_k_percent0.25-temperatureNone",
#     "cross_th0.15-top_k_percent0.5-temperature0.1",
#     "cross_th0.15-top_k_percent0.5-temperature0.3",
#     "cross_th0.15-top_k_percent0.5-temperature0.5",
#     "cross_th0.15-top_k_percent0.5-temperatureNone",
#     "cross_th0.15-top_k_percent0.75-temperature0.1",
#     "cross_th0.15-top_k_percent0.75-temperature0.3",
#     "cross_th0.15-top_k_percent0.75-temperature0.5",
#     "cross_th0.15-top_k_percent0.75-temperatureNone",
#     "cross_th0.1-top_k_percent0.25-temperature0.1",
#     "cross_th0.1-top_k_percent0.25-temperature0.3",
#     "cross_th0.1-top_k_percent0.25-temperature0.5",
#     "cross_th0.1-top_k_percent0.25-temperatureNone",
#     "cross_th0.1-top_k_percent0.5-temperature0.1",
#     "cross_th0.1-top_k_percent0.5-temperature0.3",
#     "cross_th0.1-top_k_percent0.5-temperature0.5",
#     "cross_th0.1-top_k_percent0.5-temperatureNone",
#     "cross_th0.1-top_k_percent0.75-temperature0.1",
#     "cross_th0.1-top_k_percent0.75-temperature0.3",
#     "cross_th0.1-top_k_percent0.75-temperature0.5",
#     "cross_th0.1-top_k_percent0.75-temperatureNone",
#     "cross_th0.2-top_k_percent0.25-temperature0.1",
#     "cross_th0.2-top_k_percent0.25-temperature0.3",
#     "cross_th0.2-top_k_percent0.25-temperature0.5",
#     "cross_th0.2-top_k_percent0.25-temperatureNone",
#     "cross_th0.2-top_k_percent0.5-temperature0.1",
#     "cross_th0.2-top_k_percent0.5-temperature0.3",
#     "cross_th0.2-top_k_percent0.5-temperature0.5",
#     "cross_th0.2-top_k_percent0.5-temperatureNone",
#     "cross_th0.2-top_k_percent0.75-temperature0.1",
#     "cross_th0.2-top_k_percent0.75-temperature0.3",
#     "cross_th0.2-top_k_percent0.75-temperature0.5",
#     "cross_th0.2-top_k_percent0.75-temperatureNone",
#     "cross_th0.3-top_k_percent0.25-temperature0.1",
#     "cross_th0.3-top_k_percent0.25-temperature0.3",
#     "cross_th0.3-top_k_percent0.25-temperature0.5",
#     "cross_th0.3-top_k_percent0.25-temperatureNone",
#     "cross_th0.3-top_k_percent0.5-temperature0.1",
#     "cross_th0.3-top_k_percent0.5-temperature0.3",
#     "cross_th0.3-top_k_percent0.5-temperature0.5",
#     "cross_th0.3-top_k_percent0.5-temperatureNone",
#     "cross_th0.3-top_k_percent0.75-temperature0.1",
#     "cross_th0.3-top_k_percent0.75-temperature0.3",
#     "cross_th0.3-top_k_percent0.75-temperature0.5",
#     "cross_th0.3-top_k_percent0.75-temperatureNone",
#     "cross_thmeanstd-top_k_percent0.25-temperature0.1",
#     "cross_thmeanstd-top_k_percent0.25-temperature0.3",
#     "cross_thmeanstd-top_k_percent0.25-temperature0.5",
#     "cross_thmeanstd-top_k_percent0.25-temperatureNone",
#     "cross_thmeanstd-top_k_percent0.5-temperature0.1",
#     "cross_thmeanstd-top_k_percent0.5-temperature0.3",
#     "cross_thmeanstd-top_k_percent0.5-temperature0.5",
#     "cross_thmeanstd-top_k_percent0.5-temperatureNone",
#     "cross_thmeanstd-top_k_percent0.75-temperature0.1",
#     "cross_thmeanstd-top_k_percent0.75-temperature0.3",
#     "cross_thmeanstd-top_k_percent0.75-temperature0.5",
#     "cross_thmeanstd-top_k_percent0.75-temperatureNone",
#     "cross_thmean-top_k_percent0.25-temperature0.1",
#     "cross_thmean-top_k_percent0.25-temperature0.3",
#     "cross_thmean-top_k_percent0.25-temperature0.5",
#     "cross_thmean-top_k_percent0.25-temperatureNone",
#     "cross_thmean-top_k_percent0.5-temperature0.1",
#     "cross_thmean-top_k_percent0.5-temperature0.3",
#     "cross_thmean-top_k_percent0.5-temperature0.5",
#     "cross_thmean-top_k_percent0.5-temperatureNone",
#     "cross_thmean-top_k_percent0.75-temperature0.1",
#     "cross_thmean-top_k_percent0.75-temperature0.3",
#     "cross_thmean-top_k_percent0.75-temperature0.5",
#     "cross_thmean-top_k_percent0.75-temperatureNone",
# ]

compare_list = [
    "cross_th0.05-top_k_percent0.75-temperature0.1",
    "cross_th0.05-top_k_percent0.75-temperature0.3",
    "cross_th0.05-top_k_percent0.75-temperature0.5",
    "cross_th0.05-top_k_percent0.75-temperatureNone",
    "cross_th0.15-top_k_percent0.75-temperature0.1",
    "cross_th0.15-top_k_percent0.75-temperature0.3",
    "cross_th0.15-top_k_percent0.75-temperature0.5",
    "cross_th0.15-top_k_percent0.75-temperatureNone",
    "cross_th0.1-top_k_percent0.75-temperature0.1",
    "cross_th0.1-top_k_percent0.75-temperature0.3",
    "cross_th0.1-top_k_percent0.75-temperature0.5",
    "cross_th0.1-top_k_percent0.75-temperatureNone",
    "cross_th0.2-top_k_percent0.75-temperature0.1",
    "cross_th0.2-top_k_percent0.75-temperature0.3",
    "cross_th0.2-top_k_percent0.75-temperature0.5",
    "cross_th0.2-top_k_percent0.75-temperatureNone",
    "cross_th0.3-top_k_percent0.75-temperature0.1",
    "cross_th0.3-top_k_percent0.75-temperature0.3",
    "cross_th0.3-top_k_percent0.75-temperature0.5",
    "cross_th0.3-top_k_percent0.75-temperatureNone",
    "cross_thmeanstd-top_k_percent0.75-temperature0.1",
    "cross_thmeanstd-top_k_percent0.75-temperature0.3",
    "cross_thmeanstd-top_k_percent0.75-temperature0.5",
    "cross_thmeanstd-top_k_percent0.75-temperatureNone",
    "cross_thmean-top_k_percent0.75-temperature0.1",
    "cross_thmean-top_k_percent0.75-temperature0.3",
    "cross_thmean-top_k_percent0.75-temperature0.5",
    "cross_thmean-top_k_percent0.75-temperatureNone",
]


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def parse_statistic_file(path):
    result_dict = {}
    with open(path, 'r') as fp:
        lines = islice(fp, 22, None)
        for line in lines:
            line = line.strip()
            key, value = line.split('=')
            result_dict[key] = float(value)
    return result_dict


def plot_result(viz, idx, result, update):
    for key, value in result.items():
        viz.line(X=[idx], Y=[value],
                 win=key, update="append" if update else None,
                 opts=dict(xlabel="idx", ylabel=key))


def main():
    viz = visdom.Visdom(port=8097)
    assert viz.check_connection()
    for idx, dir in enumerate(compare_list):
        sta_path = osp.join("data", "DiffuseMade_test9", "output", dir + "_ann", "statistics",
                            "dcrf-0.05-0.95", "deeplabv3*.txt")
        assert osp.isfile(sta_path), f"No such file: {sta_path}"
        result = parse_statistic_file(sta_path)
        plot_result(viz, idx, result, idx != 0)
        print(f"{idx}, {dir}")


if __name__ == '__main__':
    args = parse_args()
    main()
