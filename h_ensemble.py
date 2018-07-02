from sklearn import neural_network
from sklearn.preprocessing import LabelEncoder


class HomogeneousEnsemble():
    def __init__(self, classifier=neural_network.MLPClassifier, preprocessing_methods=[]):
        self.preprocessing_methods = preprocessing_methods
        self.number_of_classifiers = len(preprocessing_methods)

        self.classifiers = []
        self.label_encoder = None
        self.prepare_classifier_array(classifier)
        self.classifiers_weights = []
        self.classes = None

    def prepare_classifier_array(self, classifier):
        for i in range(self.number_of_classifiers):
            self.classifiers.append(classifier)

    def partial_fit(self, X, y, classes=None):
        if classes is None and self.classes is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(y)
            self.classes = self.label_encoder.classes_
            print("encoder classes: ", self.label_encoder.classes_)
        elif self.classes is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(classes)
            self.classes = classes

        print("normal y: ", y)
        y = self.label_encoder.transform(y)
        print("encoder y: ", y)

        for i in range(self.number_of_classifiers):
            resampled_X, resampled_y = self.preprocessing_methods[i].fit_sample(X, y)
            self.classifiers[i].partial_fit(resampled_X, resampled_y)

    def predict(self, X):
        prediction = 1;
        # TODO: implement
        return prediction
