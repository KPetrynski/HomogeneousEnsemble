from sklearn import neural_network

class HomogeneousEnsamble():

    def __init__(self, classifier = neural_network.MLPClassifier, preprocessing_methods = []):
        self.classifier = classifier
        self.preprocessing_methods = preprocessing_methods
        self.number_of_classifiers = len(preprocessing_methods)