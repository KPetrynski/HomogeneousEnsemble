from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn import neural_network

import streamLearn as sl


def learnMLP(X, y):
    clf = neural_network.MLPClassifier()
    clf.partial_fit(X, y)


def preprocessingTomek(self, X, y):
    smote_tomek = SMOTETomek(random_state=0)
    X_resampled, y_resampled = smote_tomek.fit_sample(X, y)
    return X_resampled, y_resampled


def run(stream_name, chunk_size, classifier=neural_network.MLPClassifier(), classifier_name="MLP"):
    streamLearner = sl.StremLearn(classifier, classifier_name, methods, methods_names, stream_name,
                                  chunk_size=chunk_size)
    streamLearner.run()


# consider also: []EditedNearestNeighbours(), CondensedNearestNeighbour(), AllKNN(), RepeatedEditedNearestNeighbours(),
# "U-ENN", "U-CNN", "U-ALLKNN", "U-RENN",

random_state = 1

methods = [RandomOverSampler(random_state=random_state), SMOTE(), ADASYN(),
           RandomUnderSampler(random_state=random_state),
           SMOTEENN(random_state=random_state), SMOTETomek(random_state=random_state)]

methods_names = ["O-ROS", "O-SMOTE", "O-ADASYN", "U-RUS", "M-SMOTEENN", "M-SMOTETOMEK"]

# stream_names = ["stream_gen_10k_0.20_1_f6_normal", "stream_gen_10k_0.20_5_f6_uniform"]
# stream_names = ["elecNormNew"]
stream_names = ["stream_gen_10k_0.20_1_f6_normal"]

chunk_size = 1000
run(stream_names[0], chunk_size)
