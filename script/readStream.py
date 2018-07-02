import numpy as np
import pandas as pd
import arff
# progress bar
import tqdm


#  Here we have X and y
def read_streams(stream_names):
    stream_name = "stream_gen_10k_0.20_1_f6_normal"
    with open('%s.arff' % stream_name, 'r') as stream:
        dataset = arff.load(stream)
        data = np.array(dataset['data'])
        dataFrame = pd.DataFrame(data)
        X = data[:, :-1].astype(np.float)
        y = data[:, -1]
        stream_size = len(y)
        number = 11
        classes = np.unique(y)
        # print("length of elements: ", stream_size, "Classes: ", classes)
        # print(X[number], " y: ", y[number])
        print_data_classes_percentage(y)
        chunkData(stream_name, X, y, stream_size)


def print_data_classes_percentage(dataset):
    unique, counts = np.unique(dataset, return_counts=True)
    data_size = len(dataset)
    print("data size: ", data_size, ", data classes: ")
    for i in range(0, len(unique)):
        print(unique[i], ": ", (counts[i] * 100) / data_size, "%")


def getChunk(X, y, start, end):
    chunk_X = X[start:end]
    chunk_y = y[start:end]
    return chunk_X, chunk_y


def chunkData(stream_name, X, y, initial_size=1000, chunk_size=500):
    print("Stream -  ", stream_name)
    for i in range(0, int(len(y)/chunk_size)):
        start = i*chunk_size
        end = start + chunk_size
        print("chunk number: ", i, " | start: ", start, " | end: ", end)
        chunk_X, chunk_y = getChunk(X, y, start, end)



# stream_names = ["stream_gen_10k_0.20_1_f6_normal", "stream_gen_10k_0.20_5_f6_uniform"]
stream_names = ["elecNormNew"]
read_streams(stream_names)
