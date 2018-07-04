import random

import arff
import numpy as np
from tqdm import tqdm


def read_streams(stream_name):
    with open('dataSets/%s.arff' % stream_name, 'r') as stream:
        dataset = arff.load(stream)
        data = np.array(dataset['data'])
        X = data[:, :-1].astype(np.float)
        y = data[:, -1]
    return X, y


def prepare_data(y):
    classes, counts = np.unique(y, return_counts=True)
    data_size = len(y)
    number_of_classes = len(classes)
    return data_size, number_of_classes, classes, counts


def print_data_classes_percentage(dataset):
    unique, counts = np.unique(dataset, return_counts=True)
    data_size = len(dataset)
    print("\ndata size: ", data_size, ", data classes: ")
    print("counts, ", counts)
    for i in range(0, len(unique)):
        print(unique[i], ": ", (counts[i] * 100) / data_size, "%")


def debalance_data(X, y, classes, counts, minority_percentage=15):
    number_of_classes = len(np.unique(y))
    data_size = len(y)
    if not number_of_classes == 2:
        print("Fail to debalance, classes: ", number_of_classes)
        return X, y
    else:
        minority, majority = [0, 1] if counts[0] < counts[1] else [1, 0]
        elements_to_delete = int(
            (counts[minority] - (counts[majority] * minority_percentage) / 100 - minority_percentage))
    print("\nStart of debalancing, data to delete: ", elements_to_delete)
    # while len(y) > data_size - elements_to_delete:
    for i in tqdm(range(elements_to_delete)):
        X, y = random_delete(X, y, classes, majority)
    print("End of debalancing")
    return X, y


def random_delete(X, y, classes, majority):
    i_delete = random.randrange(0, len(y))
    while y[i_delete] == classes[majority]:
        i_delete = random.randrange(0, len(y))
    y = np.delete(y, i_delete, axis=0)
    X = np.delete(X, i_delete, axis=0)
    return X, y


def run(dataset_name):
    X, y = read_streams(dataset_name)
    print_data_classes_percentage(y)
    data_size, number_of_classes, classes, counts = prepare_data(y)
    X, y = debalance_data(X, y, classes, counts)
    print_data_classes_percentage(y)
    return X, y


def save(X, y, stream_name):
    with open('dataSets/imbalanced%s.arff' % stream_name, 'w') as stream:
        data_string = ''
        sub_string = ''
        for i in range(len(y)):
            sub_string = ''
            for element in X[i]:
                sub_string = sub_string + str(element) + ","
            data_string = data_string + sub_string + str(y[i]) + "\n"
        stream.write(data_string)


def create_new_imbalanced_stream(name):
    X, y = run(name)
    save(X, y, name)
    print("Done")


def run_and_save(name):
    X, y = run(name)
    save(X, y, name)
    return X, y
