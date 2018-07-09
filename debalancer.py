import random
from os import listdir
from os.path import isfile, join

import arff
import numpy as np
from tqdm import tqdm


def read_streams(directory_to_read, stream_name):
    with open('%s/%s' % (directory_to_read, stream_name), 'r') as stream:
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


def debalance_data(X, y, classes, counts, minority_percentage):
    number_of_classes = len(np.unique(y))
    if not number_of_classes == 2:
        print("Fail to debalance, classes: ", number_of_classes)
        return X, y, False
    else:
        minority, majority = [0, 1] if counts[0] < counts[1] else [1, 0]
        elements_to_delete = int(
            (counts[minority] - ((counts[majority] * minority_percentage) / (100 - minority_percentage))))
    for i in tqdm(range(elements_to_delete)):
        X, y = random_delete(X, y, classes, majority)
    return X, y, True


def random_delete(X, y, classes, majority):
    i_delete = random.randrange(0, len(y))
    while y[i_delete] == classes[majority]:
        i_delete = random.randrange(0, len(y))
    y = np.delete(y, i_delete, axis=0)
    X = np.delete(X, i_delete, axis=0)
    return X, y


def run(directory_to_read, dataset_name, minority_percentage):
    X, y = read_streams(directory_to_read, dataset_name)
    data_size, number_of_classes, classes, counts = prepare_data(y)
    X, y, is_succeeded = debalance_data(X, y, classes, counts, minority_percentage)
    if is_succeeded:
        return X, y, True
    else:
        print("Error, debelances failed for dataset: ", dataset_name)
        return X, y, False


def save(X, y, directory_to_save, stream_name, minority_percentage):
    with open('%s/imb_%s_%s' % (directory_to_save, str(minority_percentage), stream_name), 'w') as stream:
        data_string = ''
        for i in range(len(y)):
            sub_string = ''
            for element in X[i]:
                sub_string = sub_string + str(element) + ","
            data_string = data_string + sub_string + str(y[i]) + "\n"
        stream.write(data_string)


def create_new_imbalanced_stream(directory_to_save, name):
    X, y = run(name)
    save(X, y, directory_to_save, name)
    print("Done")


def run_and_save(directory_to_read, directory_to_save, name, minority_percentage):
    X, y, is_succeeded = run(directory_to_read, name, minority_percentage)
    if is_succeeded:
        save(X, y, directory_to_save, name, minority_percentage)
        # return X, y


def get_files_names(directory):
    data_set_names = [f for f in listdir("%s/" % directory) if isfile(join("%s/" % directory, f))]
    print(data_set_names)
    return data_set_names


directory = "dataToDebalance"
directory_to_save = 'debalancedData'
minority_percentage_list = [33, 20, 9]
data_names = get_files_names(directory)
print("debalancer")
for name in data_names:
    for minority_percentage in minority_percentage_list:
        run_and_save(directory, directory_to_save, name, minority_percentage)
