import os

import numpy as np
import cv2
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from enum import Enum
from functools import partial
from abc import ABC, abstractmethod


def act_power(x, p=1.0):
    return np.power(x, p)


def minmax_normalize(y):
    return (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-5)


def act_tanh(x, mid=0.5, sat=4.0):
    y = 1 / (1 + np.exp(-2 * sat * (x - mid)))
    return minmax_normalize(y)


def act_tanh2(x, mid=0.5, sat=4.0):
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
    return np.array(positions)


def find_crests(hist, positions, window_size=0.1, is_trough=False, threshold=1000):
    """

    Args:
        hist:
        positions:
        window_size:
        is_trough:
        threshold:

    Returns: At least one crest/trough is returned.

    """
    crests = []
    crest_positions = []
    removed = [False] * len(hist)
    sorted_hist, sorted_positions = list(zip(*sorted(zip(hist, positions), key=lambda a: a[0], reverse=not is_trough)))
    for i in range(len(sorted_hist)):
        if threshold is not None and len(crests) and (is_trough) ^ (sorted_hist[i] - threshold < 0):
            break
        if not removed[i]:
            crests.append(sorted_hist[i])
            crest_positions.append(sorted_positions[i])
            for j in range(i + 1, len(sorted_hist)):
                if not removed[j] and abs(sorted_positions[j] - sorted_positions[i]) < window_size:
                    removed[j] = True
    return crests, crest_positions


def act_linear(x, mid=0.5):
    theta_1 = 0.5 / mid
    theta_2 = 0.5 / (1 - mid)
    beta_2 = 1 - theta_2
    up_idx = x > mid
    lo_idx = x < mid
    y = x.copy()
    y[lo_idx] = theta_1 * y[lo_idx]
    y[up_idx] = theta_2 * y[up_idx] + beta_2
    return y


def act_valley(x, vwidth=5, pprom=500, debug=False):
    hist, bin_edges = np.histogram(x, bins=100, range=(0, 1))
    positions = edge2positions(bin_edges)
    hist_smoothed2 = np.array([0] + hist.tolist())
    v_id = find_peaks(-hist, width=vwidth)[0]
    p_id = find_peaks(hist_smoothed2, prominence=pprom)[0] - 1
    if len(v_id) == 0:
        print("valleys not found!")
        valleys, valley_positions = find_crests(hist, positions, 0.2, is_trough=True, threshold=5000)
    else:
        valley_positions, valleys = positions[v_id], hist[v_id]
    if len(p_id) == 0:
        print("peaks not found!")
        peaks, peak_positions = find_crests(hist, positions, 0.2, is_trough=False, threshold=max(valleys))
    else:
        peak_positions, peaks = positions[p_id], hist[p_id]
    first_peak = min(peak_positions)
    mid = min(valley_positions)
    for p in sorted(valley_positions):
        if p > first_peak:
            mid = p
            break
    if debug:
        return act_linear(x, mid), (hist, bin_edges), (peak_positions, peaks), (valley_positions, valleys), mid
    return act_linear(x, mid)


def act_trough(x, win_size=0.1, th=1000, sat=4.0, debug=False):
    hist, bin_edges = np.histogram(x, bins=100, range=(0, 1))
    positions = edge2positions(bin_edges)
    crests, crest_positions = find_crests(hist, positions, win_size, is_trough=False, threshold=th)
    troughs, trough_positions = find_crests(hist, positions, win_size, is_trough=True, threshold=th)
    first_crest = min(crest_positions)
    mid = min(trough_positions)
    for p in sorted(trough_positions):
        if p > first_crest:
            mid = p
            break
    if debug:
        return act_tanh2(x, mid, sat), (hist, bin_edges), (crest_positions, crests), (trough_positions, troughs), mid
    return act_tanh2(x, mid, sat)


class TroughType(Enum):
    T01 = 1,
    T10 = 10,
    T101 = 101


def find_basin(merged_code):
    assert 0 < sum(merged_code) < len(merged_code)  # 0+ and 1+ is impossible
    id1 = 0
    id2 = 0
    trough_type = TroughType.T101
    for i in range(len(merged_code) - 1):
        if merged_code[i] > merged_code[i + 1]:
            id1 = i + 1
            break
    if id1 > 0:
        # 1+0+ or 1+0+1+
        find_mode101 = False
        for i in range(id1 + 1, len(merged_code)):
            if merged_code[i] == 1:
                id2 = i - 1
                find_mode101 = True
                trough_type = TroughType.T101
                break
        if not find_mode101:
            id2 = len(merged_code) - 1
            trough_type = TroughType.T10
    else:
        # 0+1+
        trough_type = TroughType.T01
        for i in range(id1 + 1, len(merged_code)):
            if merged_code[i] == 1:
                id2 = i - 1
                break
    return id1, id2, trough_type


def get_first_trough_range(sorted_crest_positions, sorted_trough_positions):
    """
    Find the sequence in the `merged_code` that matches the regular expression `10+1`.

    Args:
        sorted_crest_positions:
        sorted_trough_positions:

    Returns:

    """

    merged_code = [0] * (len(sorted_crest_positions) + len(sorted_trough_positions))
    merged_idx = [0] * (len(sorted_crest_positions) + len(sorted_trough_positions))
    cid = 0
    tid = 0
    mid = 0
    while cid < len(sorted_crest_positions) and tid < len(sorted_trough_positions):
        if sorted_crest_positions[cid] < sorted_trough_positions[tid]:
            merged_code[mid] = 1
            merged_idx[mid] = cid
            cid += 1
        else:
            merged_code[mid] = 0
            merged_idx[mid] = tid
            tid += 1
        mid += 1
    while cid < len(sorted_crest_positions):
        merged_idx[mid] = cid
        mid += 1
        cid += 1
    while tid < len(sorted_trough_positions):
        merged_idx[mid] = tid
        mid += 1
        tid += 1
    # print(merged_code)
    # print(merged_idx)
    id1, id2, trough_type = find_basin(merged_code)
    trough_idx1 = merged_idx[id1]
    trough_idx2 = merged_idx[id2]
    return trough_idx1, trough_idx2, trough_type


def act_trough2(x, win_size=0.2, th=5000, sat=4.0, debug=False):
    hist, bin_edges = np.histogram(x, bins=100, range=(0, 1))
    positions = edge2positions(bin_edges)
    troughs, trough_positions = find_crests(hist, positions, win_size, is_trough=True, threshold=th)
    crests, crest_positions = find_crests(hist, positions, win_size, is_trough=False, threshold=max(troughs))
    sorted_crest_positions = sorted(crest_positions)
    sorted_trough_positions = sorted(trough_positions)
    trough_idx1, trough_idx2, trough_type = get_first_trough_range(sorted_crest_positions, sorted_trough_positions)
    # print(f"trough_idx1={trough_idx1}, trough_idx2={trough_idx2}, trough_type={trough_type}")
    first_trough_positions = sorted_trough_positions[
                             trough_idx1:trough_idx2 + 1]
    if trough_type == TroughType.T10:
        mid = first_trough_positions[0]
    elif trough_type == TroughType.T101:
        mid = 0.5 * (first_trough_positions[0] + first_trough_positions[-1])
    else:
        mid = first_trough_positions[-1]
    if debug:
        return act_tanh2(x, mid, sat), (hist, bin_edges), (crest_positions, crests), (
            trough_positions, troughs), mid, first_trough_positions
    return act_tanh2(x, mid, sat)


def act_trough3(x, win_size=0.2, th=5000, debug=False):
    hist, bin_edges = np.histogram(x, bins=100, range=(0, 1))
    positions = edge2positions(bin_edges)
    troughs, trough_positions = find_crests(hist, positions, win_size, is_trough=True, threshold=th)
    crests, crest_positions = find_crests(hist, positions, win_size, is_trough=False, threshold=max(troughs))
    first_crest = min(crest_positions)
    mid = min(trough_positions)
    for p in sorted(trough_positions):
        if p > first_crest:
            mid = p
            break
    if debug:
        return act_linear(x, mid), (hist, bin_edges), (crest_positions, crests), (trough_positions, troughs), mid
    return act_linear(x, mid)


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


def act_he(x, sat=4.0):
    x2 = cv2.equalizeHist((x * 255).astype(np.uint8)) / 255
    return act_tanh2(x2, 0.5, sat)


class ProbActivator(ABC):
    def __init__(self):
        self.params = {}

    def add_param(self, name, value):
        self.params[name] = value
        setattr(self, name, value)

    @abstractmethod
    def do_act(self, prob):
        pass

    def __call__(self, prob):
        return self.do_act(prob)

    def __repr__(self):
        name = self.__class__.__name__[:-3].lower()
        for v in self.params.values():
            name += f"-{v}"
        return name


class DummyAct(ProbActivator):
    def __init__(self):
        super().__init__()

    def do_act(self, prob):
        return prob

    def __repr__(self):
        return ""


class LinearAct(ProbActivator):
    def __init__(self, mid):
        super().__init__()
        self.add_param("mid", mid)

    def do_act(self, prob):
        return act_linear(prob, self.mid)


class PowerAct(ProbActivator):
    def __init__(self, pow):
        super().__init__()
        self.add_param("pow", pow)

    def do_act(self, prob):
        return act_power(prob, self.pow)


class PieceAct(ProbActivator):
    def __init__(self, low, high):
        super().__init__()
        self.add_param("low", low)
        self.add_param("high", high)

    def do_act(self, prob):
        return act_piece_wise(prob, self.low, self.high)


class SoftmaxAct(ProbActivator):
    def __init__(self, mid, temp):
        super().__init__()
        self.add_param("mid", mid)
        self.add_param("temp", temp)

    def do_act(self, prob):
        return act_softmax(self.mid, self.temp)


class TanhAct(ProbActivator):
    def __init__(self, mid, sat):
        super().__init__()
        self.add_param("mid", mid)
        self.add_param("sat", sat)

    def do_act(self, prob):
        return act_tanh(prob, self.mid, self.sat)


class Tanh2Act(ProbActivator):
    def __init__(self, mid, sat):
        super().__init__()
        self.add_param("mid", mid)
        self.add_param("sat", sat)

    def do_act(self, prob):
        return act_tanh2(prob, self.mid, self.sat)


class TroughAct(ProbActivator):
    def __init__(self, win_size, th, sat):
        super().__init__()
        self.add_param("win_size", win_size)
        self.add_param("th", th)
        self.add_param("sat", sat)

    def do_act(self, prob):
        return act_trough(prob, self.win_size, self.th, self.sat)


class Trough2Act(ProbActivator):
    def __init__(self, win_size, th, sat):
        super().__init__()
        self.add_param("win_size", win_size)
        self.add_param("th", th)
        self.add_param("sat", sat)

    def do_act(self, prob):
        return act_trough2(prob, self.win_size, self.th, self.sat)


class Trough3Act(ProbActivator):
    def __init__(self, win_size, th):
        super().__init__()
        self.add_param("win_size", win_size)
        self.add_param("th", th)

    def do_act(self, prob):
        return act_trough3(prob, self.win_size, self.th)


class ValleyAct(ProbActivator):
    def __init__(self, vwidth, pprom):
        super().__init__()
        self.add_param("vwidth", vwidth)
        self.add_param("pprom", pprom)

    def do_act(self, prob):
        return act_valley(prob, self.vwidth, self.pprom)
