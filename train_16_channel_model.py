from keras.utils import plot_model

from annotation import load_processed_dataset
from dataset import ECG_generator
from models import *

if __name__ == "__main__":
    size = (4096, 16)
    epochs = 500

    diags = [15, 16, 17, 18]

    X_train, X_test, Y_train, Y_test = load_processed_dataset(diags)

    model = build_model3(size)
    plot_model(model, to_file='saved_models\\model.png')
    model.compile(loss='categorical_crossentropy', optimizer='SGD')
    model.summary()

    train_gen = ECG_generator(X_train, Y_train, 128, size[0], size[1])

    fit_generator_save(model, train_gen, 20, (X_test[:, :size[0], :size[1]], Y_test), epochs, name="model")
