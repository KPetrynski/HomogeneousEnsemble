import warnings

import numpy as np
from sklearn import neural_network, metrics
from sklearn.preprocessing import LabelEncoder


class HomogeneousEnsemble():
    def __init__(self, classifiers=None, preprocessing_methods=None,
                 weight_method=metrics.recall_score, weights_evolution_speed=0.75,
                 evaluation_weights_chunk_percentage=0.1, is_with_weights=False, neurons=100):
        if preprocessing_methods is None:
            preprocessing_methods = []
        self.preprocessing_methods = preprocessing_methods
        self.number_of_classifiers = len(preprocessing_methods)

        if classifiers is None:
            classifiers = []
        self.classifiers = classifiers
        self.classifiers_weights = []
        self.neurons = neurons
        self.classifiers_smoke_weights = []
        self.classifiers_neutral_weights = []
        self.label_encoder = None
        self.classes = None
        self.transformed_classes = None
        self.weight_method = weight_method
        self.weights_evolution_speed = weights_evolution_speed
        self.evaluation_weights_chunk_percentage = evaluation_weights_chunk_percentage

        self.prepare_classifier_array(classifiers)
        self.init_weights_array()

    def prepare_classifier_array(self, classifiers):
        if len(classifiers) < 1:
            print("initializing classifiers")
            for i in range(self.number_of_classifiers):
                self.classifiers.append(neural_network.MLPClassifier(hidden_layer_sizes=self.neurons))

    def init_weights_array(self):
        self.classifiers_weights = []
        self.classifiers_smoke_weights = []
        self.classifiers_neutral_weights = []
        for i in range(self.number_of_classifiers):
            self.classifiers_weights.append(1)
            self.classifiers_smoke_weights.append(1)
            self.classifiers_neutral_weights.append(1)

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
        self.transformed_classes = np.unique(y)
        self.learn_classifiers(X, y)

    def split_chunk(self, X, y):
        sub_chunk_size = int(len(y) * self.evaluation_weights_chunk_percentage)
        X_weight = X[0: sub_chunk_size]
        y_weight = y[0: sub_chunk_size]
        X_learn = X[sub_chunk_size:]
        y_learn = y[sub_chunk_size:]
        if not len(X_learn) == len(y_learn):
            print("----------------------- X_lern = %s, Y_learn = %s, sub_chunk_size = %s" % (
                X_learn, y_learn, sub_chunk_size))
        return X_learn, y_learn, X_weight, y_weight

    def update_weights(self, weight, i):
        self.classifiers_weights[i] = weight
        old_weight = self.classifiers_smoke_weights[i]
        new_weight = (old_weight + weight * self.weights_evolution_speed) / (1 + self.weights_evolution_speed)
        self.classifiers_smoke_weights[i] = new_weight

    def learn_classifiers(self, X, y):
        classes = np.unique(y)
        X_learn, y_learn, X_weight, y_weight = self.split_chunk(X, y)

        for i in range(self.number_of_classifiers):
            # cls = self.classifiers[i]
            try:
                resampled_X, resampled_y = self.preprocessing_methods[i].fit_sample(X_learn, y_learn)
                self.classifiers[i]._partial_fit(resampled_X, resampled_y, classes)
                weight = self.classifiers[i].score(X_weight, y_weight)
                self.update_weights(weight, i)
            except (RuntimeError, ValueError) as e:
                print("error - weight = 0.1, exception: ", e)
                weight = 0.1
                self.update_weights(weight, i)

    def get_weights(self, is_weight, is_smoke):
        weight = self.classifiers_neutral_weights
        if is_smoke:
            weight = self.classifiers_smoke_weights
        elif is_weight:
            weight = self.classifiers_weights
        return weight

    def predict_hard(self, X, is_weight, is_smoke):
        weight = self.get_weights(is_weight, is_smoke)

        predictions = np.asarray([clf.predict(X) for clf in self.classifiers]).T
        y_pred = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=weight)), axis=1,
                                     arr=predictions)
        y_pred = self.label_encoder.inverse_transform(y_pred)
        return (y_pred)

    def get_average_prediction(self, y_pred, weights):
        prediction = np.average(y_pred, weights=weights)
        return self.transformed_classes[0] if prediction > 0.5 else self.transformed_classes[1]

    def predict_soft(self, X, is_weight, is_smoke):
        weight = self.get_weights(is_weight, is_smoke)
        predictions = np.asarray([clf.predict_proba(X)[:, 0] for clf in self.classifiers]).T
        y_pred = np.apply_along_axis(lambda x: self.get_average_prediction(x, weight), axis=1,
                                     arr=predictions)
        y_pred = self.label_encoder.inverse_transform(y_pred)
        return (y_pred)

    def get_score(self, X, y):
        y_pred = self.predict_hard(X, True, True)
        return metrics.balanced_accuracy_score(y, y_pred), metrics.cohen_kappa_score(y, y_pred), \
               metrics.matthews_corrcoef(y, y_pred)

    def get_all_scores(self, X, y):
        y_soft = self.predict_soft(X, False, False)
        y_soft_w = self.predict_soft(X, True, False)
        y_soft_s = self.predict_soft(X, False, True)

        y_hard = self.predict_hard(X, False, False)
        y_hard_w = self.predict_hard(X, True, False)
        y_hard_s = self.predict_hard(X, False, True)
        predictions_y = [y_soft, y_soft_w, y_soft_s, y_hard, y_hard_w, y_hard_s]
        # print(predictions_y)
        balanced_acc_scores = []
        cohen_kappa_scores = []
        matthews_corrcoefs = []
        for y_pred in predictions_y:
            balanced_acc_scores.append(metrics.balanced_accuracy_score(y, y_pred))
            cohen_kappa_scores.append(metrics.cohen_kappa_score(y, y_pred))
            matthews_corrcoefs.append(metrics.matthews_corrcoef(y, y_pred))
        return balanced_acc_scores, cohen_kappa_scores, matthews_corrcoefs
