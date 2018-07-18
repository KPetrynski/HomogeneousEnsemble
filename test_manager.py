from os import listdir
from os.path import isfile, join

import numpy as np
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

import test_chunk_size
import test_ensemble_decision_methods
import test_number_of_neurons
import test_smoke_param


def run_chunk_size_test():
    chunks_sizes = [100, 200, 400, 600, 800, 1000, 1500, 2000, 4000]
    test_chunk_size.start(m_methods, m_methods_names, m_streams_names, chunks_sizes)


def run_number_of_neurons_test():
    # chosen chunk_size value
    chunk_size = 200
    neurons = [10, 25, 50, 75, 100, 250, 500, 750, 1000, 1500, 2000, 4000, 8000, 10000, 12500, 15000]
    test_number_of_neurons.start(m_methods, m_methods_names, m_streams_names, neurons, chunk_size)


def run_smoke_params_test():
    # define chunk_size and neurons values
    chunk_size = 200
    neurons = 750
    m_smoke_params = np.arange(0, 1.1, 0.1)
    test_smoke_param.start(m_methods, m_methods_names, m_streams_names, m_smoke_params, chunk_size, neurons)


def run_test_ensemble_decision_methods():
    print(m_streams_names)
    chunk_size = 200
    neurons = 750
    m_smoke_param = 0.75
    test_ensemble_decision_methods.start(m_methods, m_methods_names, m_streams_names, m_smoke_param, chunk_size,
                                         neurons)


def get_files_names(directory):
    data_set_names = [f for f in listdir("%s/" % directory) if isfile(join("%s/" % directory, f))]
    print(data_set_names)
    return data_set_names


m_random_state = None
m_methods = [RandomOverSampler(random_state=1), SMOTE(), ADASYN(),
             RandomUnderSampler(random_state=3),
             SMOTEENN(random_state=5), SMOTETomek(random_state=6)]
m_methods_names = ["O-ROS", "O-SMOTE", "O-ADASYN", "U-RUS", "M-SMOTEENN", "M-SMOTETOMEK"]

m_streams_names = get_files_names("debalancedData")
run_test_ensemble_decision_methods()
