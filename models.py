import tensorflow as tf
from keras.layers import *
from keras.models import Model
from sklearn.metrics import f1_score


def fit_save(model, x, y, batch_size, validation_data, epochs, name="model1"):
    Max = 0
    for i in np.arange(0, epochs):

        history = model.fit(x, y, batch_size=batch_size, validation_data=validation_data,
                            epochs=1, verbose=0)

        y_pred1 = model.predict(validation_data[0])
        y_pred = np.argmax(y_pred1, axis=1)

        y_test = np.argmax(validation_data[1], axis=1)
        tmp = f1_score(y_test, y_pred, average="macro")
        if tmp > Max:
            Max = tmp
            model.save(name + ".h5")
        print('f1 score: ' + str(tmp) + ", max: " + str(Max))


def fit_generator_save(model, generator, steps_per_epoch, validation_data, epochs, name="model1"):
    max_f1 = 0
    for epoch in np.arange(0, epochs):

        history = model.fit_generator(generator, steps_per_epoch=steps_per_epoch,
                                      epochs=1, verbose=0)

        y_prediction = np.argmax(model.predict(validation_data[0]), axis=1)
        y_labels = np.argmax(validation_data[1], axis=1)

        current_f1 = f1_score(y_labels, y_prediction, average="macro")
        if current_f1 > max_f1:
            max_f1 = current_f1
            model.save(name + ".h5")
        print('epoch: ', epoch, 'f1 score: ', current_f1, "max: ", max_f1)


class KMaxPooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """

    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

    def compute_output_shape(self, input_shape):
        return (input_shape[0], (input_shape[2] * self.k))

    def call(self, inputs):
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_input = tf.transpose(inputs, [0, 2, 1])

        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)[0]

        # return flattened output
        return top_k


def build_model1(size):
    input_ecg = Input(shape=(size, 2))

    x = Lambda(lambda layer: layer[:, :, 0:1])(input_ecg)
    x1 = Lambda(lambda layer: layer[:, :, 1:2])(input_ecg)

    for i in range(5):
        x = Conv1D(32, 5, padding='same')(x)

    x = Concatenate(axis=2)([x, x1])
    x = KMaxPooling(k=100)(x)

    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    output = Dense(2, activation="sigmoid")(x)
    model = Model(input_ecg, output)

    return model


def build_model2(size):
    input_ecg = Input(shape=(size, 2))

    x = Lambda(lambda layer: layer[:, :, 0:1])(input_ecg)
    x1 = Lambda(lambda layer: layer[:, :, 1:2])(input_ecg)

    for i in range(5):
        x = Conv1D(32, 5, dilation_rate=i + 1, padding='same')(x)

    x = Concatenate(axis=2)([x, x1])
    x = KMaxPooling(k=100)(x)

    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    output = Dense(2, activation="sigmoid")(x)
    model = Model(input_ecg, output)

    return model


def build_model3(size):
    input_ecg = Input(shape=size)

    layers_leads = []
    for i in range(size[1]):
        tmp = Lambda(lambda layer: layer[:, :, i:i + 1])(input_ecg)
        tmp = Conv1D(20, 150, strides=75)(tmp)
        layers_leads.append(Reshape((-1, 20, 1))(tmp))

    x = Concatenate(axis=3)(layers_leads)

    resid = Conv2D(7, (3, 3), dilation_rate=1, padding='same')(x)
    for i in np.arange(1, 5):
        x = Conv2D(7, (3, 3), dilation_rate=(i ** 2, i ** 2), padding='same', activation='relu')(resid)
        x = BatchNormalization()(x)
        x = Conv2D(7, (3, 3), dilation_rate=(i ** 2, i ** 2), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        resid = Concatenate(axis=3)([x, resid])

    x = Conv2D(35, (1, 1), dilation_rate=(1, 1))(resid)
    x = BatchNormalization()(x)
    x = AvgPool2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    output = Dense(2, activation="softmax")(x)
    model = Model(input_ecg, output)

    return model


def build_model4(size):
    input_ecg = Input(shape=size)

    layers_leads = []
    for i in range(size[1]):
        tmp = Lambda(lambda layer: layer[:, :, i:i + 1])(input_ecg)
        tmp = Conv1D(20, 100, strides=50)(tmp)
        layers_leads.append(Reshape((-1, 20, 1))(tmp))

    x = Concatenate(axis=3)(layers_leads)

    resid = Conv2D(7, (3, 3), dilation_rate=1, padding='same')(x)
    for i in np.arange(1, 3):
        x = Conv2D(7, (3, 3), dilation_rate=(i ** 2, i ** 2), padding='same', activation='relu')(resid)
        x = BatchNormalization()(x)
        x = Conv2D(7, (3, 3), dilation_rate=(i ** 2, i ** 2), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        resid = Concatenate(axis=3)([x, resid])

    x = Conv2D(35, (1, 1), dilation_rate=(1, 1))(resid)
    x = BatchNormalization()(x)
    x = AvgPool2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    output = Dense(2, activation="softmax")(x)
    model = Model(input_ecg, output)

    return model
