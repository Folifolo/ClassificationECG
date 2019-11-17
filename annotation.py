import pickle as pkl

import numpy as np
from keras.utils import np_utils
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from dataset import load_dataset, normalize_data


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


def load_annotation():
    with open('raw_output.pkl', 'rb') as f:
        segmentation = pkl.load(f)

    segm = np.zeros((segmentation.shape[0], segmentation.shape[1], 4))

    for i in range(4):
        segm[:, :, i] = np.where(segmentation == i, np.ones(segmentation.shape), np.zeros(segmentation.shape))

    return segm


def load_processed_dataset(diags):
    xy = load_dataset()
    X = xy["x"]
    X = normalize_data(X)
    annotation = load_annotation()
    X = np.concatenate((X, annotation), axis=2)
    Y = xy["y"]

    Y_new = np.zeros(Y.shape[0])
    for i in range(Y.shape[0]):
        for j in range(len(diags)):
            if Y[i, j] == 1:
                Y_new[i] = 1
    Y = np_utils.to_categorical(Y_new, 2)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    return X_train, X_test, Y_train, Y_test


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
