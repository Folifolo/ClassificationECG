from keras.models import Sequential, load_model, save_model, Model
from keras.utils import to_categorical
from keras.layers import *
import keras.backend as K
from keras.regularizers import l2
from keras.optimizers import RMSprop
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def fit_save(model, X, Y, batch_size, validation_data, epochs, name="model1"):
    Max = 0
    for i in np.arange(0, epochs):

        history = model.fit(X, Y, batch_size=batch_size, validation_data=validation_data,
                            epochs=1, verbose=0)

        y_pred1 = model.predict(validation_data[0])
        y_pred = np.argmax(y_pred1, axis=1)

        y_test = np.argmax(validation_data[1], axis=1)
        tmp = f1_score(y_test, y_pred, average="macro")
        if tmp > Max:
            Max = tmp
            model.save(name + ".h5")
        print('f1 score: ' + str(tmp) + ", max: " + str(Max))


def build_simple_encoder(size, reg_rate=0):
    input_ecg = Input(shape=(size, 1, 1))
    x = Conv2D(50, (90, 1), padding="same", activity_regularizer=l2(reg_rate))(input_ecg)
    x = MaxPooling2D((2, 1))(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(100, (50, 1), padding="same", activity_regularizer=l2(reg_rate))(x)
    x = MaxPooling2D((7, 1))(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(100, (30, 1), padding="same", activity_regularizer=l2(reg_rate))(x)
    x = MaxPooling2D((2, 1))(x)
    encoded = LeakyReLU(alpha=0.2)(x)

    input_encoded = Input(shape=(89, 1, 100))
    x = Conv2DTranspose(100, (30, 1), padding="same", strides=(2, 1), activity_regularizer=l2(reg_rate))(input_encoded)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2DTranspose(100, (50, 1), padding="same", strides=(7, 1), activity_regularizer=l2(reg_rate))(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2DTranspose(1, (90, 1), padding="same", strides=(2, 1), activity_regularizer=l2(reg_rate))(x)
    decoded = LeakyReLU(alpha=0.2)(x)

    encoder = Model(input_ecg, encoded, name="encoder")
    decoder = Model(input_encoded, decoded, name="decoder")
    autoencoder = Model(input_ecg, decoder(encoder(input_ecg)), name="autoencoder")

    return encoder, decoder, autoencoder


def create_dense_block(input, num_layers=4, params=[10, 50, 2]):
    for i in np.arange(0, num_layers, 1):
        if i == 0:
            x = Conv1D(params[0], params[1], padding="same")(input)
        else:
            x = Conv1D(params[0], params[1], padding="same")(inp)
        # x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        # x = MaxPooling1D(params[2])(x)
        x = ReLU()(x)
        if i == 0:
            inp = x
        else:
            tmp = MaxPool1D(params[2])(x)
            inp = Concatenate(axis=1)([inp, tmp])
    return x;


def build_simple_classifier(size, reg_rate=0):
    input_ecg = Input(shape=(size, 1))
    x = create_dense_block(input_ecg, 3, [10, 50, 2])

    x = MaxPooling1D(2)(x)
    x = Conv1D(10, 40, padding="same")(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(10, 30, padding="same")(x)

    x = create_dense_block(x, 4, [10, 20, 2])
    x = MaxPooling1D(2)(x)
    x = Conv1D(10, 10, padding="same")(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(5, 3, padding="same")(x)


    x = Flatten()(x)
    x = Dense(2024, activation="relu")(x)
    x = Dense(512, activation="relu")(x)
    output = Dense(2, activation="sigmoid")(x)
    model = Model(input_ecg, output)

    return model
