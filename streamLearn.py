import numpy as np
import pandas as pd
import arff
from tqdm import tqdm
import h_ensemble


class StremLearn():
    def __init__(self, classifier, preprocessing_methods, preprocessing_methods_names, stream_name, chunk_size = 500):
        self.ensemble = h_ensemble
        self.classifier = classifier
        self.preprocessing_methods = preprocessing_methods
        self.preprocessing_methods_names = preprocessing_methods_names
        self.stream_name = stream_name
        self.chunk_size = chunk_size

    # Here we have X and y
    def read_streams(self):
        with open('dataSets/%s.arff' % self.stream_name, 'r') as stream:
            dataset = arff.load(stream)
            data = np.array(dataset['data'])
            X = data[:, :-1].astype(np.float)
            y = data[:, -1]
        return X, y

    def print_data_classes_percentage(self, dataset):
        unique, counts = np.unique(dataset, return_counts=True)
        data_size = len(dataset)
        print("data size: ", data_size, ", data classes: ")
        for i in range(0, len(unique)):
            print(unique[i], ": ", (counts[i] * 100) / data_size, "%")

    def getChunk(self, X, y, start, end):
        chunk_X = X[start:end]
        chunk_y = y[start:end]
        return chunk_X, chunk_y

    def chunkData(self, X, y):
        number_of_samples = len(y)
        for i in tqdm(range(number_of_samples // self.chunk_size), desc='CHN', ascii=True):
            start = i * self.chunk_size
            end = start + self.chunk_size
            print("chunk number: ", i, " | start: ", start, " | end: ", end)
            chunk_X, chunk_y = self.getChunk(X, y, start, end)

            # preprocessing(chunk_X, chunk_y)

    def test_preprocessing(self, method_name, X, y):
        print("\n\n !!!! original: !!!")
        self.print_data_classes_percentage(y)
        for i in range(len(self.method_array)):
            X_resampled, y_resampled = self.method_array[i].fit_sample(X, y)
            print("--------------------------------------- \npreprocessed - ", method_name[i])
            self.print_data_classes_percentage(y_resampled)

    def run(self):
        X, y = self.read_streams()
        self.print_data_classes_percentage(y)
        self.chunkData(X, y)
