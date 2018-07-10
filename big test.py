from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn import neural_network, naive_bayes
import streamLearn as sl
import numpy as np
from os import listdir
from os.path import isfile, join
from multiprocessing.dummy import Pool as ThreadPool


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



def save_score_csv(name, neurons, balanced_acc, kappa, matthews_corrcoef, directory="results_neurons_number"):
    score_label = "number of neurons, balanced accuracy, cohen kappa, matthews corrcoef"

    score_matrix = np.stack((neurons, balanced_acc, kappa, matthews_corrcoef), axis=-1)
    file_name = '%s/aver_chunk_score_%s' % (directory, name)
    np.savetxt(file_name,
               fmt="%i, %f, %f, %f",
               header=score_label,
               X=score_matrix)


def run_for_stream(s_name, neurons, chunk_size, test_num=1, directory="results_neurons_number"):
    score_averages_balanced_acc = []
    score_averages_kappa = []
    score_averages_mathew = []
    balanced, kappa, mathew = read_and_run(s_name, chunk_size, test_num, n_neurons, methods, methods_names)
    score_averages_balanced_acc.append(balanced)
    score_averages_kappa.append(kappa)
    score_averages_mathew.append(mathew)

    return [score_averages_balanced_acc, score_averages_kappa, score_averages_mathew]

def get_files_names(directory):
    data_set_names = [f for f in listdir("%s/" % directory) if isfile(join("%s/" % directory, f))]
    print(data_set_names)
    return data_set_names


m_random_state = 1

methods = [RandomOverSampler(random_state=m_random_state), SMOTE(), ADASYN(),
           RandomUnderSampler(random_state=m_random_state),
           SMOTEENN(random_state=m_random_state), SMOTETomek(random_state=m_random_state)]

methods_names = ["O-ROS", "O-SMOTE", "O-ADASYN", "U-RUS", "M-SMOTEENN", "M-SMOTETOMEK"]

m_neuron = 100
m_test_num = 1
m_chunk_size = 1500
m_smoke_param = 0.8
m_directory = "debalancedData"
m_n_str = get_files_names(m_directory)
# pool = ThreadPool(4)
# results = pool.map(run_for_stream, n_str)
# print(results)
for name in m_n_str:
    read_and_run(name, m_neuron, m_chunk_size)
