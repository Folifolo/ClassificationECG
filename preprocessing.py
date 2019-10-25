from dataset import load_dataset
import os
import pickle as pkl
import numpy as np
from matplotlib import pyplot as plt
from  scipy.signal import medfilt
from matplotlib.pyplot import figure
from biosppy.signals import ecg


def find_peaks(x):
    peaks = []
    maximum = max(x[:, 0])

    # minimum = min(x[:, 0])
    tresh = maximum * 1 / 4
    # tresh = min(np.quantile(x[:, 0], 0.95) + maximum*1/10, maximum*1/2)
    for i in range(x.shape[0] - 1):
        if x[i, 0] > tresh:
            if x[i, 0] >= x[i + 1, 0] and x[i, 0] > x[i - 1, 0]:
                peaks.append(i)

    new_peaks = []
    for i in range(len(peaks) - 1):
        if peaks[i] < peaks[i + 1] + 10:
            # if x[peaks[i],0] > x[peaks[i]+25,0]+maximum*1/2:
            new_peaks.append(peaks[i])

    return new_peaks


def find_peaks_div(x):
    len = x.shape[0]
    y0 = np.zeros(len)
    y1 = np.zeros(len)
    y2 = np.zeros(len)
    y3 = np.zeros(len)
    for i in range(len - 2):
        y0[i + 2] = abs(x[i + 2] - x[i])
    for i in range(len - 4):
        y1[i + 4] = abs(x[i + 4] - 2 * x[i + 2] + x[i])
    for i in range(len - 4):
        y2[i + 4] = 1.3 * y0[i + 4] + 1.1 * y1[i + 4]
    for i in range(len - 4 - 7):
        for k in range(7):
            y3[i] += y2[i + 4 - k]
        y3[i] /= 8

    maxes = []

    max_curr = np.argmax(y3)
    max_curr_A = max(y3)
    maxes.append(max(0, max_curr - 10) + np.argmax(x[max(0, max_curr - 10):min(max_curr + 10, len)]))

    y3[max(0, max_curr - 50):min(max_curr + 50, len)] *= 0
    max_prev_A = max_curr_A

    max_curr = np.argmax(y3)
    max_curr_A = max(y3)

    while max_prev_A - max_curr_A < max_prev_A / 4:
        maxes.append(max(0, max_curr - 10) + np.argmax(x[max(0, max_curr - 10):min(max_curr + 10, len)]))
        y3[max(0, max_curr - 50):min(max_curr + 50, len)] *= 0
        max_prev_A = max_curr_A
        max_curr = np.argmax(y3)
        max_curr_A = max(y3)

    return maxes


def find_local_max_min(x):
    sup = []
    inf = []
    eps_max = 20
    eps_min = 10
    eps_up = 10
    eps_down = -5
    for i in range(x.shape[0]):
        tmp_max = x[i]
        tmp_min = x[i]
        for j in range(max(0, i - eps_max), min(x.shape[0], i + eps_max)):
            if tmp_max < x[j]:
                tmp_max = x[j]
                break
        if tmp_max == x[i] and tmp_max > eps_up:
            sup.append(i)

        for j in range(max(0, i - eps_min), min(x.shape[0], i + eps_min)):
            if tmp_min > x[j]:
                tmp_min = x[j]
                break
        if tmp_min == x[i] and tmp_min < eps_down:
            inf.append(i)
    return sup, inf


def cut_ecg_minmax(x, sup, inf):
    x_new = np.zeros(x.shape[0])
    x_new[sup] = x[sup]
    x_new[inf] = x[inf]
    return x_new


def make_thresh(x, sup, inf):
    x_filtred = medfilt(x, 15)
    x_new = np.zeros(x.shape[0])

    for i in sup:
        x_new[i] = x[i]
        j = i
        while x_filtred[j] >= x_filtred[min(j + 1, x.shape[0] - 1)] and x_filtred[
            min(j + 1, x.shape[0] - 1)] > 10 and j < x.shape[0] - 1:
            x_new[j + 1] = x_new[i]
            j += 1
        j = i
        while x_filtred[j] >= x_filtred[max(j - 1, 0)] and x_filtred[max(j - 1, 0)] > 10 and j > 0:
            x_new[j - 1] = x_new[i]
            j -= 1

    for i in inf:
        x_new[i] = x[i]
        j = i
        while x_filtred[j] <= x_filtred[min(j + 1, x.shape[0] - 1)] and x_filtred[
            min(j + 1, x.shape[0] - 1)] < -5 and j < x.shape[0] - 1 and x_new[j + 1] == 0:
            x_new[j + 1] = x_new[i]
            j += 1
        j = i
        while x_filtred[j] <= x_filtred[max(j - 1, 0)] and x_filtred[max(j - 1, 0)] < -5 and j > 0 and x_new[
                    j - 1] == 0:
            x_new[j - 1] = x_new[i]
            j -= 1

    return x_new


def cut_ecg_cycles(x, peaks):
    cutted_ecg = []
    cycle = []
    for peak_num in range(len(peaks) - 1):
        cycle.append(peaks[peak_num + 1] - peaks[peak_num])
        cutted_ecg.append(x[peaks[peak_num]:peaks[peak_num + 1]])
    return cutted_ecg, cycle


def scale_ecg_zeros(cutted_ecg):
    length = 250
    new_cutted = []
    for i in range(len(cutted_ecg)):
        # for ii in cutted_ecg[i].shape[1]:
        tmp = np.zeros((length, cutted_ecg[i].shape[1]))
        cur_len = cutted_ecg[i].shape[0]
        if cur_len <= 10:
            continue
        for m in range(min(cur_len // 2, length // 2)):
            tmp[m] = cutted_ecg[i][m]
        for j in range(min(cur_len - cur_len // 2, length - length // 2)):
            tmp[-j] = cutted_ecg[i][-j]

        new_cutted.append(tmp)

    return np.array(new_cutted)


def scale_ecg_reshape(cutted_ecg):
    length = 250
    new_cutted = []
    for i in range(len(cutted_ecg)):
        tmp = np.zeros((length, cutted_ecg[i].shape[1]))
        scale = length / cutted_ecg[i].shape[0]
        for j in range(length):
            tmp[j] = cutted_ecg[i][int(j // scale)]
        new_cutted.append(tmp)

    return np.array(new_cutted)


def make_mean_var(cutted_ecg):
    length = 0
    for i in range(len(cutted_ecg)):
        if cutted_ecg[i].shape[0] > length:
            length = cutted_ecg[i].shape[0]

    mean = np.zeros((length, cutted_ecg[0].shape[1]))
    var = np.zeros((length, cutted_ecg[0].shape[1]))
    for i in range(len(cutted_ecg)):
        mean += cutted_ecg[i] / len(cutted_ecg)

    for i in range(len(cutted_ecg)):
        var += ((cutted_ecg[i] - mean) * (cutted_ecg[i] - mean)) / len(cutted_ecg)
    return mean, var


def generate_normal_dataset(path='data/6002_norm.pkl'):
    xy = load_dataset()
    X = xy["x"]
    Y = xy["y"]

    new_X = np.zeros((X.shape[0], 6002))
    for i in range(X.shape[0]):
        x_i = X[i, :, :]

        peaks = ecg.ecg(signal=x_i[:, 0], sampling_rate=250., show=False)[2]

        # if len(peaks) < 2:
        #    continue
        peaks.sort()

        cytted_ecg, cycle = cut_ecg_cycles(x_i, peaks)
        cytted_ecg = scale_ecg_reshape(cytted_ecg)

        rithm = np.array(cycle)
        rithm_m = rithm.mean()
        rithm_v = rithm.std()
        rithm = np.array([rithm_m, rithm_v])
        m, v = make_mean_var(cytted_ecg)
        v = np.sqrt(v)

        new_X[i, :] = np.concatenate((np.concatenate((m, v), axis=1).flatten('F'), rithm))

    mn = new_X.mean(axis=0)
    st = new_X.std(axis=0)
    x_std = np.zeros(new_X.shape)
    for i in range(st.shape[0]):
        if st[i] == 0:
            st[i] = 1
    for i in range(X.shape[0]):
        x_std[i] = (new_X[i] - mn) / st
    output = open(path, 'wb')
    pkl.dump(new_X, output)


if __name__ == "__main__":
    generate_normal_dataset()
