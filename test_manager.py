from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
import numpy as np

import test_chunk_size
import test_number_of_neurons
import test_smoke_param
import test_ensemble_decision_methods


def run_chunk_size_test():
    # , 8000, 10000, 15000, 20000
    chunks_sizes = [100, 200, 400, 600, 800, 1000, 1500, 2000, 4000]
    test_chunk_size.start(m_methods, m_methods_names, m_streams_names, chunks_sizes)


def run_number_of_neurons_test():
    # Max value of neurons = 918, if there is more neurons, an error occurs
    # chosen chunk_size value
    chunk_size = 600
    neurons = [10, 25, 50, 75, 100, 250, 500, 750, 900]
    test_number_of_neurons.start(m_methods, m_methods_names, m_streams_names, neurons, chunk_size)


def run_smoke_params_test():
    # define chunk_size and neurons values
    chunk_size = 600
    neurons = 750
    m_smoke_params = np.arange(0, 1, 0.1)
    test_smoke_param.start(m_methods, m_methods_names, m_streams_names, m_smoke_params, chunk_size, neurons)


def run_test_ensemble_decision_methods():
    chunk_size = 600
    neurons = 750
    m_smoke_param = 0.75
    test_ensemble_decision_methods.start(m_methods, m_methods_names, m_streams_names, m_smoke_param, chunk_size,
                                         neurons)


m_random_state = 1
m_methods = [RandomOverSampler(random_state=m_random_state), SMOTE(), ADASYN(),
             RandomUnderSampler(random_state=m_random_state),
             SMOTEENN(random_state=m_random_state), SMOTETomek(random_state=m_random_state)]
m_methods_names = ["O-ROS", "O-SMOTE", "O-ADASYN", "U-RUS", "M-SMOTEENN", "M-SMOTETOMEK"]
# m_streams_names = ["imb_9_sd_s_hyp_r1_s_hyp_r3.arff", "imb_33_sd_s_hyp_r1_s_hyp_r3.arff",
#                    "imb_9_sd_s_rbf_r1_s_rbf_r3.arff", "imb_33_sd_s_rbf_r1_s_rbf_r3.arff"]
# m_streams_names = ["imb_20_sd_s_rbf_r1_s_rbf_r3.arff", "imb_20_sd_s_hyp_r1_s_hyp_r3.arff"]

m_streams_names = ["imb_9_s_hyp_r3_small.arff", "imb_20_s_hyp_r2_small.arff"]
run_smoke_params_test()
