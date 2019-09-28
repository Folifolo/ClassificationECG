from models import build_simple_classifier, f1
from dataset import load_dataset, get_diag_dict, normalize_data
import pickle as pkl
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from keras.optimizers import RMSprop
from keras.models import Sequential, load_model, save_model, Model
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# from keras.utils import plot_model

if __name__ == "__main__":
    xy = load_dataset()
    X = xy["x"]
    Y = xy["y"]
    X = normalize_data(X[:])
    X = X[:, :, :1]
    Y = Y[:, 0:1]

    size = 2492

    Y_new = np.zeros((Y.shape[0], 2))
    for i in range(len(Y)):
        if Y[i]:
            Y_new[i, 0] = 1
        else:
            Y_new[i, 1] = 1
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_new, test_size=0.33, random_state=42)
    model = build_simple_classifier(size)

    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=[f1])

    # plot_model(model, to_file='model.png')
    model.summary()
    Max = 0
    for i in np.arange(0, 1000):

        history = model.fit(X_train[:, :size], Y_train, batch_size=128, validation_data=(X_test[:, :size], Y_test),
                            epochs=1, verbose=0)

        y_pred1 = model.predict(X_test[:, :size])
        y_pred = np.argmax(y_pred1, axis=1)

        y_test = np.argmax(Y_test, axis=1)
        tmp = f1_score(y_test, y_pred, average="macro")
        if tmp > Max:
            Max = tmp
            model.save("modelq" + str(i))
        print(str(tmp) + " " + str(Max))
