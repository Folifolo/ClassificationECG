import pickle as pkl
import numpy as np
from dataset import load_dataset
from matplotlib import pyplot as plt
from scipy import signal

if __name__ == "__main__":

    xy = load_dataset()
    X = xy["x"]

    width = 128
    widths = np.arange(1, width + 1)
    new_X = np.zeros((X.shape[0], width, X.shape[1]))
    i_prev = 0
    for i in range(len(X)):
        cwtmatr = signal.cwt(X[i, :, 0], signal.ricker, widths)
        new_X[i] = cwtmatr
        print('epoch: ', i)
        if (i - 1) % 100 == 99:
            output = open('data\\spectre' + str(i // 100) + '.pkl', 'wb')
            pkl.dump(new_X[i_prev:i], output)
            i_prev = i

    output = open('data\\spectre.pkl', 'wb')
    pkl.dump(new_X[i_prev:], output)
