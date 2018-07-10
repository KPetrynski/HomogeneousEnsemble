from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

import test_chunk_size
import test_number_of_neurons
import test_smoke_param


def run_chunk_size_test():
    test_chunk_size.start(m_methods, m_methods_names, m_streams_names)


def run_number_of_neurons_test():
    # Max value of neurons = 918, if there is more neurons, an error occurs
    neurons = [10, 25, 50, 75, 100, 250, 500, 750, 900]
    # define chunk_size value
    test_number_of_neurons.start(m_methods, m_methods_names, m_streams_names, neurons)


def run_smoke_params_test():
    # define chunk_size and neurons values
    m_smoke_params = [0.25, 0.5, 0.75, 0.8, 1]
    test_smoke_param.start(m_methods, m_methods_names, m_streams_names, m_smoke_params)


m_random_state = 1
m_methods = [RandomOverSampler(random_state=m_random_state), SMOTE(), ADASYN(),
             RandomUnderSampler(random_state=m_random_state),
             SMOTEENN(random_state=m_random_state), SMOTETomek(random_state=m_random_state)]
m_methods_names = ["O-ROS", "O-SMOTE", "O-ADASYN", "U-RUS", "M-SMOTEENN", "M-SMOTETOMEK"]
m_streams_names = ["imb_9_s_hyp_r2.arff", "imb_9_s_rbf_r2.arff", "imb_20_sd_s_rbf_r1_s_rbf_r3.arff",
                   "imb_20_sd_s_hyp_r1_s_hyp_r3.arff"]
