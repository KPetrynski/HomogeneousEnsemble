import numpy as np

import streamLearn as sl


def save_scores_csv(scores_acc, scores_kappa, scores_matthews_corrcoef, stream_range, s_name,
                    directory_to_save, neurons):
    score_label = "elements, balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef"
    balanced_acc = np.asarray(scores_acc)
    kappa = np.asarray(scores_kappa)
    matthews_corrcoef = np.asarray(scores_matthews_corrcoef)

    score_matrix = np.stack((stream_range, balanced_acc, kappa, matthews_corrcoef), axis=-1)
    file_name = '%s/%s_neurons_%s' % (directory_to_save, s_name, str(neurons))
    np.savetxt(file_name,
               X=score_matrix,
               fmt="%i, %f, %f, %f",
               header=score_label)


def save_score_csv_average(s_name, neurons, balanced_acc, kappa, matthews_corrcoef,
                           directory="results_neurons_number"):
    score_label = "neurons, balanced accuracy, cohen kappa, matthews corrcoef"

    score_matrix = np.stack((neurons, balanced_acc, kappa, matthews_corrcoef), axis=-1)
    file_name = '%s/av_neurons_%s' % (directory, s_name)
    np.savetxt(file_name,
               X=score_matrix,
               fmt="%i, %f, %f, %f",
               header=score_label)


def read_and_run(stream_name, chunk_size, prep_methods, prep_methods_names, neurons=100, smoke_param=1,
                 classifier_name="MLP", directory_to_save="results_neurons_number"):
    streamLearner = sl.StremLearn(None, classifier_name, prep_methods, prep_methods_names, stream_name,
                                  smoke_weight_param=smoke_param, chunk_size=chunk_size, number_of_neurons=neurons,
                                  score_name="hard_score")
    streamLearner.read_and_run()
    print("get scores")
    balanced_acc, kappa, matthews_corrcoef, stream_range = streamLearner.get_scores()
    save_scores_csv(balanced_acc, kappa, matthews_corrcoef, stream_range, stream_name, directory_to_save, neurons)
    return streamLearner.get_score_averages()


def run_for_stream_neurons(s_name, methods, methods_names, chunk_size, neurons_numbers):
    print("run_for_stream_neurons")
    score_averages_balanced_acc = []
    score_averages_kappa = []
    score_averages_mathew = []
    for neurons_number in neurons_numbers:
        print("neurons_number: ", neurons_number)
        balanced_acc, kappa, mathew = read_and_run(s_name, chunk_size, methods, methods_names, neurons_number)
        score_averages_balanced_acc.append(balanced_acc)
        score_averages_kappa.append(kappa)
        score_averages_mathew.append(mathew)

    save_score_csv_average(s_name, neurons_numbers, score_averages_balanced_acc, score_averages_kappa,
                           score_averages_mathew)


def start(m_methods, m_methods_names, m_streams_names, m_neurons, m_chunk_size=200):
    # Max value of neurons = 918, if there is more neurons, an error occurs
    for m_name in m_streams_names:
        run_for_stream_neurons(m_name, m_methods, m_methods_names, m_chunk_size, m_neurons)
