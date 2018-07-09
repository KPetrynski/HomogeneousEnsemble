from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn import neural_network, naive_bayes
import streamLearn as sl
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool


def learnMLP(X, y):
    clf = neural_network.MLPClassifier()
    clf.partial_fit(X, y)


def preprocessingTomek(self, X, y):
    smote_tomek = SMOTETomek(random_state=0)
    X_resampled, y_resampled = smote_tomek.fit_sample(X, y)
    return X_resampled, y_resampled


def run(stream_name, chunk_size, test_number, neurons_in_layer, prep_methods, prep_methods_names, m_X, m_y,
        classifier_name="MLP"):
    classifier = neural_network.MLPClassifier(hidden_layer_sizes=(neurons_in_layer,))

    streamLearner = sl.StremLearn(classifier, classifier_name, prep_methods, prep_methods_names, stream_name,
                                  test_number,
                                  chunk_size=chunk_size, number_of_neurons=neurons_in_layer)
    streamLearner.run(m_X, m_y)


def read_and_run(stream_name, chunk_size, test_number, neurons_in_layer, prep_methods, prep_methods_names,
                 classifier_name="MLP"):
    classifier = neural_network.MLPClassifier(hidden_layer_sizes=(neurons_in_layer,))

    streamLearner = sl.StremLearn(classifier, classifier_name, prep_methods, prep_methods_names, stream_name,
                                  test_number,
                                  chunk_size=chunk_size, number_of_neurons=neurons_in_layer)
    streamLearner.read_and_run()
    return streamLearner.get_score_averages()


def save_score_csv(name, chunk_sizes, balanced_acc, kappa, matthews_corrcoef, directory="results_chunk_size"):
    # score_label = "chunk_sizes, balanced accuracy, cohen kappa, matthews corrcoef"
    score_label = "number of neurons, balanced accuracy, cohen kappa, matthews corrcoef"

    score_matrix = np.stack((chunk_sizes, balanced_acc, kappa, matthews_corrcoef), axis=-1)
    file_name = '%s/aver_chunk_score_%s' % (directory, name)
    np.savetxt(file_name,
               fmt="%i, %f, %f, %f",
               header=score_label,
               X=score_matrix)


def run_for_stream_chunk(s_name):
    score_averages_balanced_acc = []
    score_averages_kappa = []
    score_averages_mathew = []
    for chunk_size in chunks_sizes:
        balanced, kappa, mathew = read_and_run(s_name, chunk_size, test_num, neurons, methods, methods_names)
        score_averages_balanced_acc.append(balanced)
        score_averages_kappa.append(kappa)
        score_averages_mathew.append(mathew)
    save_score_csv(s_name, chunks_sizes, score_averages_balanced_acc, score_averages_kappa, score_averages_mathew)
    return [score_averages_balanced_acc, score_averages_kappa, score_averages_mathew]


def save_score_csv_neurons(name, neurons, balanced_acc, kappa, matthews_corrcoef, directory="results_neurons_number"):
    score_label = "number of neurons, balanced accuracy, cohen kappa, matthews corrcoef"

    score_matrix = np.stack((neurons, balanced_acc, kappa, matthews_corrcoef), axis=-1)
    file_name = '%s/aver_chunk_score_%s' % (directory, name)
    np.savetxt(file_name,
               fmt="%i, %f, %f, %f",
               header=score_label,
               X=score_matrix)


def run_for_stream_neurons(s_name, neurons, chunk_size, test_num=1, directory="results_neurons_number"):
    score_averages_balanced_acc = []
    score_averages_kappa = []
    score_averages_mathew = []
    for n_neurons in neurons:
        balanced, kappa, mathew = read_and_run(s_name, chunk_size, test_num, n_neurons, methods, methods_names)
        score_averages_balanced_acc.append(balanced)
        score_averages_kappa.append(kappa)
        score_averages_mathew.append(mathew)
    save_score_csv_neurons(s_name, neurons, score_averages_balanced_acc, score_averages_kappa, score_averages_mathew,
                           directory)
    return [score_averages_balanced_acc, score_averages_kappa, score_averages_mathew]


# consider also: []EditedNearestNeighbours(), CondensedNearestNeighbour(), AllKNN(), RepeatedEditedNearestNeighbours(),
# "U-ENN", "U-CNN", "U-ALLKNN", "U-RENN",

m_random_state = 1

methods = [RandomOverSampler(random_state=m_random_state), SMOTE(), ADASYN(),
           RandomUnderSampler(random_state=m_random_state),
           SMOTEENN(random_state=m_random_state), SMOTETomek(random_state=m_random_state)]

methods_names = ["O-ROS", "O-SMOTE", "O-ADASYN", "U-RUS", "M-SMOTEENN", "M-SMOTETOMEK"]

m_stream_names = ["s_hyp_r1", "s_hyp_r2", "s_rbf_r2", "id_s_hyp_r2_s_hyp_r3",
                  "sd_s_hyp_r2_s_rbf_r2"]
# chunk_small = 400
# chunk_big = 500
# chunk_step = 100
# Max value of neurons = 918, if there is more neurons, an error occurs
m_neurons = [10, 50, 100, 250, 500, 750, 918]
# m_neurons = [918, 1000, 1500, 5000, 10000]
m_test_num = 1
m_chunk_size = 1500
# m_chunks_sizes = [250, 500, 750, 1000, 1500, 2000, 3000, 4000, 4500, 5000, 5500, 7000]
# neurons_sizes = [50, 100, 500]



# "imb_9_sd_s_rbf_r1_s_rbf_r3", "imb_20_sd_s_rbf_r1_s_rbf_r3","imb_33_sd_s_rbf_r1_s_rbf_r3", "imb_9_sd_s_hyp_r1_s_hyp_r3", "imb_20_sd_s_hyp_r1_s_hyp_r3",
m_n_str = ["imb_20_sd_s_rbf_r1_s_rbf_r3", "imb_20_sd_s_hyp_r1_s_hyp_r3"]
# pool = ThreadPool(4)
# results = pool.map(run_for_stream, n_str)
# print(results)
for name in m_n_str:
    run_for_stream_neurons(name, m_neurons, m_chunk_size)
