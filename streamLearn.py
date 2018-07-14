import arff
import numpy as np

import h_ensemble
import utils as ut
from tqdm import tqdm


class StremLearn():
    def __init__(self, classifiers, classifier_name, preprocessing_methods, preprocessing_methods_names, stream_name,
                 init_chunk=1000, prediction_step=750, smoke_weight_param=1, chunk_size=500, number_of_neurons=100,
                 directory="testDataSet", step=50, is_with_weights=False):
        self.ensemble = h_ensemble.HomogeneousEnsemble(classifiers, preprocessing_methods,
                                                       weights_evolution_speed=smoke_weight_param,
                                                       is_with_weights=False, neurons=number_of_neurons)
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
        self.init_chunk = init_chunk
        self.stream_range_prediction = []
        self.stream_range_learning = []
        self.prediction_step = prediction_step
        self.step = step

    def read_streams(self):
        with open('%s/%s' % (self.directory, self.stream_name), 'r') as stream:
            dataset = arff.load(stream)
            data = np.array(dataset['data'])
            X = data[:, :-1].astype(np.float)
            y = data[:, -1]
        return X, y

    def first_training(self, X, y):
        chunk_X, chunk_y = ut.get_chunk(X, y, 0, self.init_chunk)
        self.ensemble.partial_fit(chunk_X, chunk_y)

    def test_and_train(self, X, y):
        number_of_samples = len(y)
        self.first_training(X, y)
        accumulation_prediction = 0
        accumulation_learning = 0
        stream_range_calculations = np.arange(self.init_chunk, number_of_samples + 1, self.step)

        for element in tqdm(stream_range_calculations):
            if accumulation_prediction >= self.prediction_step:
                self.stream_range_prediction.append(element)
                start = element - self.prediction_step
                end = element
                chunk_X, chunk_y = ut.get_chunk(X, y, start, end)
                score_acc, score_kappa, score_matthews_corrcoef = self.ensemble.get_all_scores(chunk_X, chunk_y)
                self.scores_acc.append(score_acc)
                self.scores_kappa.append(score_kappa)
                self.scores_matthews_corrcoef.append(score_matthews_corrcoef)
                accumulation_prediction = 0

            if accumulation_learning >= self.chunk_size:
                start = element - self.chunk_size
                end = element
                chunk_X, chunk_y = ut.get_chunk(X, y, start, end)
                self.ensemble.partial_fit(chunk_X, chunk_y)
                accumulation_learning = 0
            accumulation_prediction += self.step
            accumulation_learning += self.step

    def get_scores(self):
        return self.scores_acc, self.scores_kappa, self.scores_matthews_corrcoef, self.stream_range_prediction

    def get_all_score_averages(self):
        balanced_acc = []
        kappa = []
        matthews_corrcoef = []
        print("get_all_score_averages acc: ", self.scores_acc)
        for column in range(np.asarray(self.scores_acc).shape[1]):
            balanced_acc.append(np.average(np.array(self.scores_acc)[:, column]))
            kappa.append(np.average(np.array(self.scores_kappa)[:, column]))
            matthews_corrcoef.append(np.average(np.array(self.scores_matthews_corrcoef)[:, column]))
        return balanced_acc, kappa, matthews_corrcoef

    def print_scores(self):
        print("Classifier: ", self.classifier_name, ", Chunk size: ", self.chunk_size)
        print("accuracy_score: ", self.scores_acc)
        print("cohen_kappa_score: ", self.scores_kappa)
        print("scores_matthews_corrcoef ", self.scores_matthews_corrcoef)

    def read_and_run(self):
        self.ensemble.init_weights_array()
        X, y = self.read_streams()
        # self.print_data_classes_percentage(y)
        self.test_and_train(X, y)
