import arff
import numpy as np
from tqdm import tqdm

import h_ensemble
from utils import get_chunk


class StremLearn():
    def __init__(self, classifier, classifier_name, preprocessing_methods, preprocessing_methods_names, stream_name,
                 smoke_weight_param=1, chunk_size=500, number_of_neurons=100, directory="debalancedData"):
        self.ensemble = h_ensemble.HomogeneousEnsemble(classifier, preprocessing_methods,
                                                       weights_evolution_speed=smoke_weight_param)
        self.classifier = classifier
        self.classifier_name = classifier_name
        self.preprocessing_methods = preprocessing_methods
        self.preprocessing_methods_names = preprocessing_methods_names
        self.stream_name = stream_name
        self.chunk_size = chunk_size
        self.scores_acc = []
        self.scores_kappa = []
        self.scores_matthews_corrcoef = []
        self.number_of_neurons = number_of_neurons
        self.smoke_weight_param = smoke_weight_param
        self.directory = directory

    def read_streams(self):
        with open('%s/%s' % (self.directory, self.stream_name), 'r') as stream:
            dataset = arff.load(stream)
            data = np.array(dataset['data'])
            X = data[:, :-1].astype(np.float)
            y = data[:, -1]
        return X, y

    def first_training(self, X, y):
        chunk_X, chunk_y = get_chunk(X, y, 0, self.chunk_size)
        self.ensemble.partial_fit(chunk_X, chunk_y)

    def test_and_train(self, X, y):
        number_of_samples = len(y)
        self.first_training(X, y)
        for i in tqdm(range(1, number_of_samples // self.chunk_size), desc='CHN', ascii=True):
            start = i * self.chunk_size
            end = start + self.chunk_size
            chunk_X, chunk_y = get_chunk(X, y, start, end)
            score_acc, score_kappa, score_matthews_corrcoef = self.ensemble.get_score(X, y)
            self.scores_acc.append(score_acc)
            self.scores_kappa.append(score_kappa)
            self.scores_matthews_corrcoef.append(score_matthews_corrcoef)
            self.ensemble.partial_fit(chunk_X, chunk_y)

    def get_scores(self):
        return self.scores_acc, self.scores_kappa, self.scores_matthews_corrcoef

    def get_score_averages(self):
        balanced_acc = np.average(self.scores_acc)
        kappa = np.average(self.scores_kappa)
        matthews_corrcoef = np.average(self.scores_matthews_corrcoef)
        return balanced_acc, kappa, matthews_corrcoef

    def print_scores(self):
        print("Classifier: ", self.classifier_name, ", Chunk size: ", self.chunk_size)
        print("accuracy_score: ", self.scores_acc)
        print("cohen_kappa_score: ", self.scores_kappa)
        print("scores_matthews_corrcoef ", self.scores_matthews_corrcoef)

    def read_and_run(self):
        self.ensemble.reset()
        X, y = self.read_streams()
        # self.print_data_classes_percentage(y)
        self.test_and_train(X, y)
