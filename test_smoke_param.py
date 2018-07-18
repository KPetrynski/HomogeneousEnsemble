import numpy as np

import streamLearn as sl


def save_scores_csv(scores_acc, scores_kappa, scores_matthews_corrcoef, stream_range, s_name,
                    directory_to_save, smoke_param):
    score_label = "elements, balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef"
    balanced_acc = np.asarray(scores_acc)
    kappa = np.asarray(scores_kappa)
    matthews_corrcoef = np.asarray(scores_matthews_corrcoef)

    score_matrix = np.stack((stream_range, balanced_acc, kappa, matthews_corrcoef), axis=-1)
    file_name = '%s/%s_smoke_param_%s' % (directory_to_save, s_name, str(smoke_param))
    np.savetxt(file_name,
               X=score_matrix,
               fmt="%i, %f, %f, %f",
               header=score_label)


def save_score_csv_average(s_name, smoke_params, balanced_acc, kappa, matthews_corrcoef,
                           directory="results_smoke_weights"):
    score_label = "smoke param, balanced accuracy, cohen kappa, matthews corrcoef"

    score_matrix = np.stack((smoke_params, balanced_acc, kappa, matthews_corrcoef), axis=-1)
    file_name = '%s/av_smoke_params_%s' % (directory, s_name)
    np.savetxt(file_name,
               X=score_matrix,
               fmt="%f, %f, %f, %f",
               header=score_label)


def read_and_run(stream_name, chunk_size, prep_methods, prep_methods_names, neurons=750, smoke_param=1,
                 classifier_name="MLP", directory_to_save="results_smoke_weights", ):
    streamLearner = sl.StremLearn(None, classifier_name, prep_methods, prep_methods_names, stream_name,
                                  smoke_weight_param=smoke_param, chunk_size=chunk_size, number_of_neurons=neurons,
                                  is_with_weights=True, score_name="hard_smoke_score")
    streamLearner.read_and_run()
    balanced_acc, kappa, matthews_corrcoef, stream_range = streamLearner.get_scores()
    save_scores_csv(balanced_acc, kappa, matthews_corrcoef, stream_range, stream_name, directory_to_save, smoke_param)
    return streamLearner.get_score_averages()


def run_for_stream_smoke_weights(s_name, methods, methods_names, chunk_size, neurons, smoke_params):
    print("run_for_stream_smoke_weights")
    score_averages_balanced_acc = []
    score_averages_kappa = []
    score_averages_mathew = []
    for smoke_param in smoke_params:
        print("smoke_param: ", smoke_param)
        balanced_acc, kappa, mathew = read_and_run(s_name, chunk_size, methods, methods_names, neurons, smoke_param)
        score_averages_balanced_acc.append(balanced_acc)
        score_averages_kappa.append(kappa)
        score_averages_mathew.append(mathew)
    print("save")
    save_score_csv_average(s_name, smoke_params, score_averages_balanced_acc, score_averages_kappa,
                           score_averages_mathew)


# Max value of neurons = 918, if there is more neurons, an error occurs
def start(m_methods, m_methods_names, m_streams_names, m_smoke_params, m_chunk_size=200, m_neurons=750):
    for m_name in m_streams_names:
        run_for_stream_smoke_weights(m_name, m_methods, m_methods_names, m_chunk_size, m_neurons, smoke_params=m_smoke_params)
