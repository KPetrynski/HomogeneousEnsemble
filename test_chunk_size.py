import numpy as np
from sklearn import neural_network

import streamLearn as sl


def save_scores_csv(scores_acc, scores_kappa, scores_matthews_corrcoef, stream_range, chunk_size, s_name,
                    directory_to_save):
    score_label = "elements, balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef"
    balanced_acc = np.asarray(scores_acc)
    kappa = np.asarray(scores_kappa)
    matthews_corrcoef = np.asarray(scores_matthews_corrcoef)

    score_matrix = np.stack((stream_range, balanced_acc, kappa, matthews_corrcoef), axis=-1)
    file_name = '%s/%s_chunk_size_%s' % (directory_to_save, s_name, str(chunk_size))
    np.savetxt(file_name,
               X=score_matrix,
               fmt="%i, %f, %f, %f",
               header=score_label)


def save_score_csv_average(s_name, chunk_sizes, balanced_acc, kappa, matthews_corrcoef,
                           directory="results_chunk_size"):
    score_label = "Chunk size, balanced accuracy, cohen kappa, matthews corrcoef"

    score_matrix = np.stack((chunk_sizes, balanced_acc, kappa, matthews_corrcoef), axis=-1)
    file_name = '%s/av_chunk_score_%s' % (directory, s_name)
    np.savetxt(file_name,
               X=score_matrix,
               fmt="%i, %f, %f, %f",
               header=score_label)


def read_and_run(stream_name, chunk_size, prep_methods, prep_methods_names, neurons=100, smoke_param=1,
                 classifier_name="MLP", directory_to_save="results_chunk_size"):
    classifier = neural_network.MLPClassifier(hidden_layer_sizes=neurons)

    streamLearner = sl.StremLearn(classifier, classifier_name, prep_methods, prep_methods_names, stream_name,
                                  smoke_weight_param=smoke_param, chunk_size=chunk_size, number_of_neurons=neurons)
    streamLearner.read_and_run()
    balanced_acc, kappa, matthews_corrcoef, stream_range = streamLearner.get_scores()
    save_scores_csv(balanced_acc, kappa, matthews_corrcoef, stream_range, chunk_size, stream_name, directory_to_save)
    return streamLearner.get_score_averages()


def run_for_stream_chunk(s_name, methods, methods_names, chunks_sizes):
    print("run_for_stream_chunk")
    score_averages_balanced_acc = []
    score_averages_kappa = []
    score_averages_mathew = []
    for chunk_size in chunks_sizes:
        print("chunk_size: ", chunk_size)
        balanced_acc, kappa, mathew = read_and_run(s_name, chunk_size, methods, methods_names)
        score_averages_balanced_acc.append(balanced_acc)
        score_averages_kappa.append(kappa)
        score_averages_mathew.append(mathew)

    save_score_csv_average(s_name, chunks_sizes, score_averages_balanced_acc, score_averages_kappa,
                           score_averages_mathew)


def start(m_methods, m_methods_names, m_streams_names, m_chunks_sizes):
    for m_name in m_streams_names:
        run_for_stream_chunk(m_name, m_methods, m_methods_names, m_chunks_sizes)
