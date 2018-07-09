import arff
import numpy as np
from tqdm import tqdm

import h_ensemble


class StremLearn():
    def __init__(self, classifier, classifier_name, preprocessing_methods, preprocessing_methods_names, stream_name,
                 test_number,
                 chunk_size=500, number_of_neurons=100):
        self.ensemble = h_ensemble.HomogeneousEnsemble(classifier, preprocessing_methods)
        self.classifier = classifier
        self.classifier_name = classifier_name
        self.preprocessing_methods = preprocessing_methods
        self.preprocessing_methods_names = preprocessing_methods_names
        self.stream_name = stream_name
        self.chunk_size = chunk_size
        self.scores_acc = []
        self.scores_kappa = []
        self.scores_matthews_corrcoef = []
        self.test_number = test_number
        self.number_of_neurons = number_of_neurons

    # Here we have X and y
    def read_streams(self):
        with open('debalancedData/%s.arff' % self.stream_name, 'r') as stream:
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

    def first_training(self, X, y):
        chunk_X, chunk_y = self.getChunk(X, y, 0, self.chunk_size)
        self.ensemble.partial_fit(chunk_X, chunk_y)

    def test_and_train(self, X, y):
        number_of_samples = len(y)
        self.first_training(X, y)
        for i in tqdm(range(1, number_of_samples // self.chunk_size), desc='CHN', ascii=True):
            start = i * self.chunk_size
            end = start + self.chunk_size
            chunk_X, chunk_y = self.getChunk(X, y, start, end)
            score_acc, score_kappa, score_matthews_corrcoef = self.ensemble.get_score(X, y)
            self.scores_acc.append(score_acc)
            self.scores_kappa.append(score_kappa)
            self.scores_matthews_corrcoef.append(score_matthews_corrcoef)
            self.ensemble.partial_fit(chunk_X, chunk_y)
        # self.print_scores()
        self.save_score_csv()
        a, b, c = self.get_score_averages()
        # self.save_scores()

    def prepare_score_string(self):
        scores_string = '' + self.stream_name + ', chunk size: ' + str(self.chunk_size) + ', neurons: ' \
                        + str(self.number_of_neurons) + "\n"
        scores_string = scores_string + 'KAPPA, ' + str(self.scores_kappa) + '\n'
        scores_string = scores_string + 'ACC, ' + str(self.scores_acc) + '\n'
        return scores_string

    def save_score_csv(self):
        score_label = "elements, balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef"
        balanced_acc = np.asarray(self.scores_acc)
        kappa = np.asarray(self.scores_kappa)
        matthews_corrcoef = np.asarray(self.scores_matthews_corrcoef)
        range = np.arange(self.chunk_size, (len(balanced_acc) + 1) * self.chunk_size, self.chunk_size)
        score_matrix = np.stack((range, balanced_acc, kappa, matthews_corrcoef), axis=-1)
        file_name = 'results_neurons_number/%s_neurons_%s' %(self.stream_name, str(self.number_of_neurons))
        np.savetxt(file_name,
                   fmt="%i, %f, %f, %f",
                   header=score_label,
                   X=score_matrix)

    def get_score_averages(self):
        balanced_acc = np.average(self.scores_acc)
        kappa = np.average(self.scores_kappa)
        matthews_corrcoef = np.average(self.scores_matthews_corrcoef)
        # print("average: ", balanced_acc, kappa, matthews_corrcoef)
        return balanced_acc, kappa, matthews_corrcoef

    def save_scores(self):
        score_string = self.prepare_score_string()
        test_name = 'res_' + self.stream_name + str(self.chunk_size) + str(self.number_of_neurons)
        with open('results/%s.arff' % test_name, 'w') as stream:
            stream.write(score_string)

    def test_preprocessing(self, X, y):
        print("\n\n !!!! original: !!!")
        self.print_data_classes_percentage(y)
        for i in range(len(self.preprocessing_methods)):
            X_resampled, y_resampled = self.preprocessing_methods[i].fit_sample(X, y)
            print("--------------------------------------- \npreprocessed - ", self.preprocessing_methods_names[i])
            self.print_data_classes_percentage(y_resampled)

    def print_scores(self):
        print("Classifier: ", self.classifier_name, ", Chunk size: ", self.chunk_size)
        print("accuracy_score: ", self.scores_acc)
        print("cohen_kappa_score: ", self.scores_kappa)
        print("scores_matthews_corrcoef ", self.scores_matthews_corrcoef)

    def run(self, m_X, m_y):
        self.ensemble.reset()
        # X, y = self.read_streams()
        X = m_X
        y = m_y
        self.print_data_classes_percentage(y)
        self.test_and_train(X, y)

    def read_and_run(self):
        self.ensemble.reset()
        X, y = self.read_streams()
        # self.print_data_classes_percentage(y)
        self.test_and_train(X, y)
