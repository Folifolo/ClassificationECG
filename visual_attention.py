import innvestigate.utils
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt

from annotation import load_processed_dataset

if __name__ == "__main__":

    size = (4096, 16)

    diags = [15, 16, 17, 18]

    _, x_test, _, y_test = load_processed_dataset(diags)

    model = load_model('saved_models\\model.h5')

    for i in range(len(model.layers)):
        model.layers[i].name = "layer" + str(i)

    model.summary()

    prediction = np.argmax(model.predict(x_test[:, :size[0], :size[1]]), axis=1)
    label = np.argmax(y_test, axis=1)

    model = innvestigate.utils.model_wo_softmax(model)

    analyzer = innvestigate.create_analyzer("input_t_gradient", model, allow_lambda_layers=True)

    for index in np.arange(0, x_test.shape[0]):
        a = analyzer.analyze(x_test[index:index + 1, :size[0], :size[1]])

        a /= np.max(np.abs(a))
        a = a[0, :]

        x_axis = np.arange(size[0])
        plt.plot(x_axis, x_test[index, :size[0], 0], color='black')
        plt.plot(x_axis, a - 1, color='red', alpha=1)

        plt.title('diag: ' + str(label[index]) + ', pred: ' + str(prediction[index]))
        plt.show()
