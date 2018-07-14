import numpy as np
from sklearn import neural_network

import streamLearn as sl


# def save_scores_csv(scores_acc, scores_kappa, scores_matthews_corrcoef, stream_range, s_name,
#                     directory_to_save, smoke_param):
#     score_label = "elements, balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef"
#     balanced_acc = np.asarray(scores_acc)
#     kappa = np.asarray(scores_kappa)
#     matthews_corrcoef = np.asarray(scores_matthews_corrcoef)
#
#     score_matrix = np.stack((stream_range, balanced_acc, kappa, matthews_corrcoef), axis=-1)
#     file_name = '%s/%s_smoke_param_%s' % (directory_to_save, s_name, str(smoke_param))
#     np.savetxt(file_name,
#                X=score_matrix,
#                fmt="%i, %f, %f, %f",
#                header=score_label)


def save_score_csv_average(s_name, results, directory="results", subdirectory="balanced_acc"):
    score_label = "stream_name, soft, soft weighted, soft smoke weighted, hard, hard weighted, hard smoke weighted"
    print("save to: ", directory, "/", subdirectory)
    print(results)
    results = np.array(results)
    score_matrix = np.stack(
        (s_name, results[:, 0], results[:, 1], results[:, 2], results[:, 3], results[:, 4], results[:, 5]), axis=-1)
    # results[:, 0], results[:, 1], results[:, 2], results[:, 3], results[:, 4], results[:, 5]
    print("czo ta maciez: \n", score_matrix)
    file_name = '%s/%s_av_all' % (directory, subdirectory)
    np.savetxt(file_name,
               X=score_matrix,
               fmt="%i, %f, %f, %f, %f, %f, %f",
               header=score_label)


def read_and_run(stream_name, chunk_size, prep_methods, prep_methods_names, neurons=750, smoke_param=0.75,
                 classifier_name="MLP", ):
    streamLearner = sl.StremLearn(None, classifier_name, prep_methods, prep_methods_names, stream_name,
                                  smoke_weight_param=smoke_param, chunk_size=chunk_size, number_of_neurons=neurons,
                                  is_with_weights=True, score_name="all_scores")
    streamLearner.read_and_run()
    # balanced_acc, kappa, matthews_corrcoef, stream_range = streamLearner.get_scores()
    # save_scores_csv(balanced_acc, kappa, matthews_corrcoef, stream_range, stream_name, directory_to_save, smoke_param)
    return streamLearner.get_all_score_averages()


# Max value of neurons = 918, if there is more neurons, an error occurs
def start(m_methods, m_methods_names, m_streams_names, m_smoke_params=0.75, m_chunk_size=600, m_neurons=750):
    score_balanced_acc = []
    score_kappa = []
    score_matthews_corrcoefs = []

    for m_name in m_streams_names:
        sc_balanced_acc, sc_kappa, sc_matthews = read_and_run(m_name, m_chunk_size, m_methods, m_methods_names,
                                                              m_neurons, m_smoke_params)
        print("staart acc ", sc_balanced_acc)
        score_balanced_acc.append(sc_balanced_acc)
        score_kappa.append(sc_kappa)
        score_matthews_corrcoefs.append(sc_matthews)

    print("time to save")
    a = [1, 2]
    save_score_csv_average(a, score_balanced_acc, subdirectory="balanced_acc")
    save_score_csv_average(a, score_kappa, subdirectory="kappa")
    save_score_csv_average(a, score_matthews_corrcoefs, subdirectory="matthews")
    # run_for_stream_smoke_weights(m_name, m_methods, m_methods_names, m_chunk_size, m_neurons, m_smoke_params)
