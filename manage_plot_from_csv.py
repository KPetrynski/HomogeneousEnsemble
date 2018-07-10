import plot_from_csv
from os import listdir
from os.path import isfile, join


def get_files_names(directory):
    data_set_names = [f for f in listdir("%s/" % directory) if isfile(join("%s/" % directory, f))]
    print(data_set_names)
    return data_set_names


# method_number = 1
directory = "results_smoke_weights/average_score"
title = ["", "average balanced accuracy", "average cohen kappa", "average matthews corrcoef"]
score_method_name = ["", "balanced accuracy", "cohen kappa", "matthews corrcoef"]
x_label = "data stream"
# y_label = score_method_name[method_number] + " score"
names = get_files_names(directory)
# names = ["aver_chunk_score_imb_20_sd_s_rbf_r1_s_rbf_r3"]
for method_number in range(1, 2):
    y_label = score_method_name[method_number] + " score"
    plot_from_csv.plot_compare_streams(names, directory, title=title[method_number], x_label=x_label, y_label=y_label,
                                       method=method_number)
