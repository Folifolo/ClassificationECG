from keras.models import Sequential, load_model, save_model, Model
from keras.utils import to_categorical
from keras.layers import *
import keras.backend as K
from keras.regularizers import l2
from keras.optimizers import RMSprop


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


def build_simple_classifier(size, reg_rate=0):
    num_layers = 3
    alpha = 0.2
    layers = [
        [32, 10, 4],
        [32, 20, 4],
        [32, 30, 4]
    ]
    input_ecg = Input(shape=(size, 1))
    for i in np.arange(0, num_layers, 1):
        layer_params = layers[i]
        if i == 0:
            x = Conv1D(layer_params[0], layer_params[1], padding="same")(input_ecg)
        else:
            x = Conv1D(layer_params[0], layer_params[1], padding="same")(x)
        x = Dropout(0.2)(x)
        x = Conv1D(layer_params[0], layer_params[1], padding="same")(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(layer_params[2])(x)
        x = ReLU()(x)

    x = Flatten()(x)
    x = Dense(2048, activation="relu")(x)
    x = Dense(512, activation="relu")(x)
    output = Dense(2, activation="sigmoid")(x)
    model = Model(input_ecg, output)

    return model
