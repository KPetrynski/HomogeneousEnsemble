import numpy as np

import streamLearn as sl



def save_score_csv_average(s_name, results, directory="results", subdirectory="balanced_acc"):
    score_label = "stream_name, soft, soft weighted, soft smoke weighted, hard, hard weighted, hard smoke weighted"
    print("save to: ", directory, "/", subdirectory)
    print(results)
    results = np.array(results)
    score_matrix = np.stack(
        (s_name, results[:, 0], results[:, 1], results[:, 2], results[:, 3], results[:, 4], results[:, 5]), axis=-1)
    file_name = '%s/%s_av_all' % (directory, subdirectory)
    np.savetxt(file_name,
               X=score_matrix,
               fmt="%s",
               header=score_label)


def read_and_run(stream_name, chunk_size, prep_methods, prep_methods_names, neurons=750, smoke_param=0.75,
                 classifier_name="MLP", ):
    streamLearner = sl.StremLearn(None, classifier_name, prep_methods, prep_methods_names, stream_name,
                                  smoke_weight_param=smoke_param, chunk_size=chunk_size, number_of_neurons=neurons,
                                  is_with_weights=True, score_name="all_scores")
    streamLearner.read_and_run()
    return streamLearner.get_all_score_averages()



def start(m_methods, m_methods_names, m_streams_names, m_smoke_params=0.75, m_chunk_size=600, m_neurons=750):
    score_balanced_acc = []
    score_kappa = []
    score_matthews_corrcoefs = []
    s_names = []

    for m_name in m_streams_names:
        sc_balanced_acc, sc_kappa, sc_matthews = read_and_run(m_name, m_chunk_size, m_methods, m_methods_names,
                                                              m_neurons, m_smoke_params)
        score_balanced_acc.append(sc_balanced_acc)
        score_kappa.append(sc_kappa)
        score_matthews_corrcoefs.append(sc_matthews)
        s_names.append(str(m_name[:-5]))

    print("time to save")
    a = [1, 2, 3]
    save_score_csv_average(s_names, score_balanced_acc, subdirectory="balanced_acc")
    save_score_csv_average(s_names, score_kappa, subdirectory="kappa")
    save_score_csv_average(s_names, score_matthews_corrcoefs, subdirectory="matthews")

