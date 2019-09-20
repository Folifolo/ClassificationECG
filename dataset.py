import BaselineWanderRemoval as bwr
import pickle as pkl
import numpy as np
import json
import sys
import os

folder_path = "C:\\data\\"
data_file_name = "data_2033.json"
diag_file_name = "diagnosis.json"
pkl_file_name = "data_2033.pkl"

leads_names = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
FREQUENCY_OF_DATASET = 250


def parser(folder_path):
    try:
        infile = open(folder_path + data_file_name, 'rb')
        data = json.load(infile)
        diag_dict = get_diag_dict()

        X = []
        Y = []
        for id in data.keys():

            leads = data[id]['Leads']
            diagnosis = data[id]['StructuredDiagnosisDoc']

            y = []
            try:
                for diag in diag_dict.keys():
                    y.append(diagnosis[diag])
            except KeyError:
                print("\nThe patient " + id + " is not included in the final dataset. Reason: no diagnosis.")
                continue
            y = np.where(y, 1, 0)

            x = []
            try:
                for lead in leads_names:
                    rate = int(leads[lead]['SampleRate'] / FREQUENCY_OF_DATASET)
                    x.append(leads[lead]['Signal'][::rate])
            except KeyError:
                print("\nThe patient " + id + " is not included in the final dataset. Reason: no lead.")
                continue

            X.append(x)
            Y.append(y)

        X = np.array(X)
        Y = np.array(Y)
        X = np.swapaxes(X, 1, 2)

        print("The dataset is parsed.")
        print("X shape: ", X.shape)
        print("Y shape: ", Y.shape)

        return {"x": X, "y": Y}

    except FileNotFoundError:
        print("File " + data_file_name + " has not found.\nThe specified folder (" + folder_path +
              ") must contain files with data (" + data_file_name +
              ") and file with structure of diagnosis (" + diag_file_name + ").")
        sys.exit(0)


def get_diag_dict():
    def deep(data, diag_list):
        for diag in data:
            if diag['type'] == 'diagnosis':
                diag_list.append(diag['name'])
            else:
                deep(diag['value'], diag_list)

    try:
        infile = open(folder_path + diag_file_name, 'rb')
        data = json.load(infile)

        diag_list = []
        deep(data, diag_list)

        diag_num = list(range(len(diag_list)))
        diag_dict = dict(zip(diag_list, diag_num))

        return diag_dict

    except FileNotFoundError:
        print("File " + diag_file_name + " has not found.\nThe specified folder (" + folder_path +
              ") must contain files with data (" + data_file_name +
              ") and file with structure of diagnosis (" + diag_file_name + ").")
        sys.exit(0)


def load_dataset(folder_path=folder_path):
    if not os.path.exists(folder_path + pkl_file_name):
        xy = parser(folder_path)
        fix_bw(xy, folder_path)

    infile = open(folder_path + pkl_file_name, 'rb')
    dataset = pkl.load(infile)
    infile.close()

    return dataset


def fix_bw(xy, folder_path):
    print("Baseline wondering fixing is started. It's take some time.")

    X = xy["x"]
    patients_num = X.shape[0]
    for i in range(patients_num):
        print("\rSignal %s/" % str(i + 1) + str(patients_num) + ' is fixed.', end='')
        for j in range(X.shape[2]):
            X[i, :, j] = bwr.fix_baseline_wander(X[i, :, j], FREQUENCY_OF_DATASET)
    xy['x'] = X

    outfile = open(folder_path + pkl_file_name, 'wb')
    pkl.dump(xy, outfile)
    outfile.close()

    print("The dataset is saved.")


def normalize_data(X):
    mn = X.mean(axis=0)
    st = X.std(axis=0)
    x_std = np.zeros(X.shape)
    for i in range(X.shape[0]):
        x_std[i] = (X[i] - mn) / st
    return x_std


if __name__ == "__main__":
    xy = load_dataset()
