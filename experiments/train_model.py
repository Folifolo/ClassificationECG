import numpy as np
from keras.utils import plot_model
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from dataset import load_dataset, normalize_data
from models import *
from test_models import *

if __name__ == "__main__":
    xy = load_dataset()
    X = xy["x"]
    Y = xy["y"]
    X = normalize_data(X[:])
    X = X[:, :, :1]
    Y = Y[:, 0:1]

    size = 1024

    Y_new = np.zeros((Y.shape[0], 2))
    for i in range(len(Y)):
        if Y[i]:
            Y_new[i, 0] = 1
        else:
            Y_new[i, 1] = 1
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_new, test_size=0.33, random_state=42)
    model = build_model1(size)
    # model = load_model("model1.h5")
    model.compile(loss='categorical_crossentropy', optimizer='SGD')

    plot_model(model, to_file='model.png')
    # model = load_model("modelq19", custom_objects={'f1': f1})
    model.summary()
    fit_save(model, X_train[:, :size], Y_train, batch_size=128, validation_data=(X_test[:, :size], Y_test),
             epochs=100, name="model2")
