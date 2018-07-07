from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn import neural_network, naive_bayes
import debalancer
import streamLearn as sl


def learnMLP(X, y):
    clf = neural_network.MLPClassifier()
    clf.partial_fit(X, y)


def preprocessingTomek(self, X, y):
    smote_tomek = SMOTETomek(random_state=0)
    X_resampled, y_resampled = smote_tomek.fit_sample(X, y)
    return X_resampled, y_resampled


def run(stream_name, chunk_size, test_number, neurons_in_layer, prep_methods, prep_methods_names,  m_X, m_y,
        classifier_name="MLP"):
    classifier = neural_network.MLPClassifier(hidden_layer_sizes=(neurons_in_layer,))

    streamLearner = sl.StremLearn(classifier, classifier_name, prep_methods, prep_methods_names, stream_name,
                                  test_number,
                                  chunk_size=chunk_size, number_of_neurons=neurons_in_layer)
    streamLearner.run(m_X, m_y)


# consider also: []EditedNearestNeighbours(), CondensedNearestNeighbour(), AllKNN(), RepeatedEditedNearestNeighbours(),
# "U-ENN", "U-CNN", "U-ALLKNN", "U-RENN",

random_state = 1

methods = [RandomOverSampler(random_state=random_state), SMOTE(), ADASYN(),
           RandomUnderSampler(random_state=random_state),
           SMOTEENN(random_state=random_state), SMOTETomek(random_state=random_state)]

methods_names = ["O-ROS", "O-SMOTE", "O-ADASYN", "U-RUS", "M-SMOTEENN", "M-SMOTETOMEK"]

stream_names = ["s_hyp_r1", "s_hyp_r2", "s_rbf_r2", "id_s_hyp_r2_s_hyp_r3",
                "sd_s_hyp_r2_s_rbf_r2"]
chunk_small = 400
chunk_big = 500
chunk_step = 100
neurons = 100
test_num = 1
chunks_sizes = [500, 1000, 2500]
neurons_sizes = [50, 100, 500]

# for chunk_size in range(chunk_small, chunk_big, chunk_step):
#     run(stream_name=stream_names[1], chunk_size=1000, classifier=classifiers[0],
#         classifier_name=classifiers_names[i])
for stream_name in stream_names:
    X, y = debalancer.run_and_save(stream_name)
    for my_chunk_size in chunks_sizes:
        for neurons in neurons_sizes:
            run(stream_name, my_chunk_size, test_num, neurons, methods, methods_names,  X, y)

