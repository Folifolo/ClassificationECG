import torch
from dataset import load_dataset
from matplotlib import pyplot as plt
import pickle as pkl
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# import torchvision


def smooth_annotator_output(signal, tolerance=20):
    for i in range(signal.shape[0]):
        interval_size = 0
        prev_qrs_end = 0
        is_prev_complex_qrs = False

        for j in np.arange(1, signal.shape[1] - 1):

            if signal[i, j] == 2:

                if signal[i, j - 1] == 0:
                    if is_prev_complex_qrs:
                        is_prev_complex_qrs = False
                        interval_size = j - prev_qrs_end
                        if interval_size < tolerance:
                            signal[i, prev_qrs_end:j] = np.ones(interval_size)

                if signal[i, j + 1] == 0:
                    is_prev_complex_qrs = True
                    prev_qrs_end = j

            elif signal[i, j] != 0:
                is_prev_complex_qrs = False
    return signal


def get_qrs_intervals(signal):
    intervals = []

    for i in range(signal.shape[0]):
        tmp = []
        start = end = 0
        for j in range(signal.shape[1] - 2):

            if signal[i, j] == 2:

                if signal[i, j - 1] != 2:
                    start = j
                elif signal[i, j + 1] != 2:
                    end = j
                    tmp.append([start, end])

        intervals.append(tmp)

    return intervals


def get_r_peaks(ecg, intervals):
    R_peaks = []

    for i in range(ecg.shape[0]):
        tmp = []
        for j in intervals[i]:
            maximum = np.argmax((ecg[i, j[0]:j[1]])) + j[0]
            tmp.append(maximum)
        R_peaks.append(tmp)

    return R_peaks


if __name__ == "__main__":

    path = 'data/6002_norm.pkl'

    with open(path, 'rb') as f:
        data = pkl.load(f)
    print(data.shape)
    plt.plot(data[0])
    plt.show()

    xy = load_dataset()
    X = xy["x"]
    Y = xy["y"]

    infile = open('tmp.pkl', 'rb')
    dataset = pkl.load(infile)
    infile.close()

    dataset = smooth_annotator_output(dataset, 20)

    intervals = get_qrs_intervals(dataset)

    R_peaks = get_r_peaks(X[:, :, 0], intervals)

    new_X = np.zeros((X.shape[0], 6002))

    from preprocessing import *

    for i in range(X.shape[0]):
        x_i = X[i, :, :]

        cutted_ecg, cycle = cut_ecg_cycles(x_i, R_peaks[i])
        cutted_ecg = scale_ecg_reshape(cutted_ecg)

        rithm = np.array(cycle)
        rithm_m = rithm.mean()
        rithm_v = rithm.std()
        rithm = np.array([rithm_m, rithm_v])
        m, v = make_mean_var(cutted_ecg)
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
    pkl.dump(x_std, output)
