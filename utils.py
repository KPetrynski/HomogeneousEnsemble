@staticmethod
def print_data_classes_percentage(dataset):
    unique, counts = np.unique(dataset, return_counts=True)
    data_size = len(dataset)
    print("data size: ", data_size, ", data classes: ")
    for i in range(0, len(unique)):
        print(unique[i], ": ", (counts[i] * 100) / data_size, "%")


@staticmethod
def get_chunk(X, y, start, end):
    chunk_X = X[start:end]
    chunk_y = y[start:end]
    return chunk_X, chunk_y