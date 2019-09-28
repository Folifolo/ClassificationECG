from keras.models import Sequential, load_model, save_model, Model
from keras.utils import to_categorical
from keras.layers import *
import keras.backend as K
from keras.regularizers import l2
from keras.optimizers import RMSprop
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


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


def create_dence_block(input, num_layers=4, params=[10, 50, 4]):
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
            inp = Concatenate(axis=1)([inp, x])
    return x;


def build_simple_classifier(size, reg_rate=0):
    input_ecg = Input(shape=(size, 1))

    x = create_dence_block(input_ecg, 4, [10, 50, 4])

    x = MaxPooling1D(4)(x)
    x = Conv1D(10, 50, padding="same")(x)
    x = MaxPooling1D(4)(x)
    x = Conv1D(10, 50, padding="same")(x)

    x = create_dence_block(x, 4, [10, 50, 4])
    x = MaxPooling1D(4)(x)
    x = Conv1D(10, 50, padding="same")(x)
    x = MaxPooling1D(4)(x)
    x = Conv1D(10, 50, padding="same")(x)

    x = Flatten()(x)
    x = Dense(2048, activation="relu")(x)
    x = Dense(512, activation="relu")(x)
    output = Dense(2, activation="sigmoid")(x)
    model = Model(input_ecg, output)

    return model
