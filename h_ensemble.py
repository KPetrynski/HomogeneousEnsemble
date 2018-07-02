from sklearn import neural_network, metrics, model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import warnings


class HomogeneousEnsemble():
    def __init__(self, classifier=neural_network.MLPClassifier(), preprocessing_methods=[], weight_method = metrics.recall_score):
        self.preprocessing_methods = preprocessing_methods
        self.number_of_classifiers = len(preprocessing_methods)

        self.classifiers = []
        self.classifiers_weights = []
        self.prepare_classifier_array(classifier)
        self.label_encoder = None
        self.classes = None
        self.weight_method = weight_method


    def prepare_classifier_array(self, classifier):
        for i in range(self.number_of_classifiers):
            self.classifiers.append(classifier)
            self.classifiers_weights.append(0.5)

    def partial_fit(self, X, y, classes=None):
        warnings.filterwarnings(action='ignore', category=DeprecationWarning)
        if classes is None and self.classes is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(y)
            self.classes = self.label_encoder.classes_
            print("encoder classes: ", self.label_encoder.classes_)
        elif self.classes is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(classes)
            self.classes = classes

        # print("normal y: ", y)
        y = self.label_encoder.transform(y)
        # print("encoder y: ", y)
        self.learn_classifiers(X, y)

        # self.predict(X, y)

    def learn_classifiers(self, X, y):
        for i in range(self.number_of_classifiers):
            cls = self.classifiers[i]
            resampled_X, resampled_y = self.preprocessing_methods[i].fit_sample(X, y)
            classes = np.unique(y)
            cls._partial_fit(resampled_X, resampled_y, classes)
            y_pred = cls.predict(X)
            weight = cls.score(X, y)
            self.classifiers_weights[i] = weight
