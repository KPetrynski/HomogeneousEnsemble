import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read(name):
    with open("results/%s.arff" % name, 'r') as result_file:
        content = []
        for line in result_file:
            content.append(line)
    return content


def getHeaderValues(content):
    header = content.split(', ')
    chunk = header[1].split(': ')
    neuron = header[2].split(': ')
    header_name = header[0]
    chunk_size = int(chunk[1])
    neuron_value = int(neuron[1].split()[0])
    return header_name, chunk_size, neuron_value


def getResultArray(content):
    start = '['
    end = ']'
    results = content[content.find(start) + len(start):content.rfind(end)]
    results = results.split(', ')
    results_array = []
    for element in results:
        results_array.append(float(element))
    return results_array


def getScoresFromFile(name):
    content = read(name)
    header_name, chunk_size, neuron_value = getHeaderValues(content[0])
    kappa_score = getResultArray(content[1])
    acc_score = getResultArray(content[2])
    # print(header_name, chunk_size, neuron_value, kappa_score, acc_score)
    return header_name, chunk_size, neuron_value, kappa_score, acc_score


def plotResults(name):
    header_name, chunk_size, neuron_value, kappa_score, acc_score = getScoresFromFile(name)
    tilte = "stream: " + header_name + "Chunk size: " + str(chunk_size) + " number of chunks: " + str(len(
        kappa_score)) + "number of neurons: " + str(neuron_value)
    # print(header_name)
    print("length", len(kappa_score))
    x, y = range(len(kappa_score)), kappa_score
    plt.plot(x, y, zorder=1)
    plt.scatter(x, y)

    plt.title(tilte)
    plt.xlabel("chunk")
    plt.ylabel("kappa score")
    plt.show()


# file_name = "res_s_hyp_r150050"
# file_name = "res_s_hyp_r1500100"
# file_name = "res_s_hyp_r1500500"
file_name = "res_s_hyp_r22500100"

plotResults(file_name)
