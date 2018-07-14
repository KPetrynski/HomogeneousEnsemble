import warnings

import numpy as np
from sklearn import neural_network, metrics
from sklearn.preprocessing import LabelEncoder


class HomogeneousEnsemble():
    def __init__(self, classifiers=None, preprocessing_methods=[],
                 weight_method=metrics.recall_score, weights_evolution_speed=1,
                 evaluation_weights_chunk_percentage=0.1, is_with_weights=False, neurons=100):
        self.preprocessing_methods = preprocessing_methods
        self.number_of_classifiers = len(preprocessing_methods)

        self.classifiers = classifiers
        self.classifiers_weights = []
        self.prepare_classifier_array(classifiers)
        self.label_encoder = None
        self.classes = None
        self.weight_method = weight_method
        self.scores_acc = []
        self.scores_kappa = []
        self.scores_matthews_corrcoef = []
        self.weights_evolution_speed = weights_evolution_speed
        self.evaluation_weights_chunk_percentage = evaluation_weights_chunk_percentage
        self.is_with_weights = is_with_weights
        self.neurons = neurons

    def reset(self):
        self.scores_kappa = []
        self.scores_acc = []
        self.scores_matthews_corrcoef = []

    def prepare_classifier_array(self, classifiers):
        if classifiers is None:
            for i in range(self.number_of_classifiers):
                self.classifiers.append(neural_network.MLPClassifier(hidden_layer_sizes=self.neurons))

    def init_weights_array(self):
        for i in range(self.number_of_classifiers):
            self.classifiers_weights.append(1)

    def partial_fit(self, X, y, classes=None):
        warnings.filterwarnings(action='ignore', category=DeprecationWarning)
        if classes is None and self.classes is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(y)
            self.classes = self.label_encoder.classes_
            # print("encoder classes: ", self.label_encoder.classes_)
        elif self.classes is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(classes)
            self.classes = classes

        y = self.label_encoder.transform(y)
        self.learn_classifiers(X, y)

    def split_chunk(self, X, y):
        sub_chunk_size = int(len(y) * self.evaluation_weights_chunk_percentage)
        X_weight = X[0: sub_chunk_size]
        y_weight = y[0: sub_chunk_size]
        X_learn = X[sub_chunk_size:]
        y_learn = y[sub_chunk_size:]
        if not len(X_learn) == len(y_learn):
            print("----------------------- X_lern = %s, Y_learn = %s, sub_chunk_size = %s" % (X_learn, y_learn, sub_chunk_size))
        return X_learn, y_learn, X_weight, y_weight

    def update_weights(self, weight, i, is_with_weights):
        if self.is_with_weights:
            old_weight = self.classifiers_weights[i]
            new_weight = (old_weight + weight * self.weights_evolution_speed) / (1 + self.weights_evolution_speed)
            self.classifiers_weights[i] = new_weight

    def learn_classifiers(self, X, y):
        classes = np.unique(y)
        X_learn, y_learn, X_weight, y_weight = self.split_chunk(X, y)

        for i in range(self.number_of_classifiers):
            cls = self.classifiers[i]
            try:
                resampled_X, resampled_y = self.preprocessing_methods[i].fit_sample(X_learn, y_learn)
                cls._partial_fit(resampled_X, resampled_y, classes)
                if self.is_with_weights:
                    weight = cls.score(X_weight, y_weight)
                    self.update_weights(weight, i)
            except (RuntimeError, ValueError) as e:
                print("error - weight = 0.1, exception: ", e)
                if self.is_with_weights:
                    weight = 0.1
                    self.update_weights(weight, i)

    def predict(self, X):
        predictions = np.asarray([clf.predict(X) for clf in self.classifiers]).T
        y_pred = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=self.classifiers_weights)), axis=1,
                                     arr=predictions)
        y_pred = self.label_encoder.inverse_transform(y_pred)
        return (y_pred)

    def get_score(self, X, y):
        y_pred = self.predict(X)
        return metrics.balanced_accuracy_score(y, y_pred), metrics.cohen_kappa_score(y, y_pred), \
               metrics.matthews_corrcoef(y, y_pred)

    def get_final_scores(self):
        return self.scores_acc, self.scores_kappa
