import numpy as np
import pandas as pd
import arff
from imblearn.combine import SMOTEENN, SMOTETomek
# progress bar
from tqdm import tqdm
from sklearn import neural_network
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours, CondensedNearestNeighbour, AllKNN, RepeatedEditedNearestNeighbours

#  Here we have X and y
def read_streams(stream_names):
    for stream_name in stream_names:
        print("\nStream -  ", stream_name)
        with open('dataSets/%s.arff' % stream_name, 'r') as stream:
            dataset = arff.load(stream)
            data = np.array(dataset['data'])
            X = data[:, :-1].astype(np.float)
            y = data[:, -1]
            stream_size = len(y)
            print_data_classes_percentage(y)
            chunkData(stream_name, X, y, stream_size)


def print_data_classes_percentage(dataset):
    unique, counts = np.unique(dataset, return_counts=True)
    data_size = len(dataset)
    print("data size: ", data_size, ", data classes: ")
    for i in range(0, len(unique)):
        print(unique[i], ": ", (counts[i] * 100) / data_size, "%")


def getChunk(X, y, start, end):
    chunk_X = X[start:end]
    chunk_y = y[start:end]
    return chunk_X, chunk_y


def chunkData(stream_name, X, y, initial_size=1000, chunk_size=2500):
    number_of_samples = len(y)
    for i in tqdm(range(number_of_samples // chunk_size), desc='CHN', ascii=True):
        start = i * chunk_size
        end = start + chunk_size
        # print("chunk number: ", i, " | start: ", start, " | end: ", end)
        chunk_X, chunk_y = getChunk(X, y, start, end)
        preprocessing(chunk_X, chunk_y)


def learnMLP(X, y):
    clf = neural_network.MLPClassifier()
    clf.partial_fit(X, y)

def preprocessing(X, y):
    smote_tomek = SMOTETomek(random_state=0)
    X_resampled, y_resampled = smote_tomek.fit_sample(X, y)
    return X_resampled, y_resampled

def init_preprocessing_methods(random_state = 1):
    return [RandomOverSampler(random_state=random_state), SMOTE(), ADASYN(),
                     RandomUnderSampler(random_state=random_state), EditedNearestNeighbours(random_state=random_state),
                     CondensedNearestNeighbour(random_state=random_state), AllKNN(random_state=random_state),
                     RepeatedEditedNearestNeighbours(random_state=random_state),
                     SMOTEENN(random_state=random_state), SMOTETomek(random_state=random_state)]

# stream_names = ["stream_gen_10k_0.20_1_f6_normal", "stream_gen_10k_0.20_5_f6_uniform"]
# stream_names = ["elecNormNew"]
stream_names = ["stream_gen_10k_0.20_1_f6_normal"]
read_streams(stream_names)
