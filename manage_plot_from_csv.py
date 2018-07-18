import plot_from_csv
from os import listdir
from os.path import isfile, join


def get_files_names(directory):
    data_set_names = [f for f in listdir("%s/" % directory) if isfile(join("%s/" % directory, f))]
    print(data_set_names)
    return data_set_names


# ------------------ --average
# frame = [0.3, 0.8]
# width_param = 10
# height_param = 10
# legend_column = 2
# grid_x_step = 0.2
# grid_x_step_min = 0.1
# x_label = "Smoke parameter"

# --------------------- stream
frame = [0.4, 1]
width_param = 20
height_param = 10
legend_column = 3
grid_x_step = 10000
grid_x_step_min = 1000
x_label = "data stream"

sub_directory = "drift"
end_name = "drift"
directory_from = "results_neurons_number/" + sub_directory
directory_to = "results_neurons_number_plots/" + sub_directory
title = ["", "average_balanced_accuracy", "average_cohen_kappa", "average_matthews_corrcoef"]
title = ["", "average_balanced_accuracy", "average_cohen_kappa", "average_matthews_corrcoef"]
score_method_name = ["", "balanced accuracy", "cohen kappa", "matthews corrcoef"]

# y_label = score_method_name[method_number] + " score"
names = get_files_names(directory_from)
# names = ["aver_chunk_score_imb_20_sd_s_rbf_r1_s_rbf_r3"]
for method_number in range(1, 4):
    y_label = score_method_name[method_number] + " score"
    plot_from_csv.plot_compare_streams(names, directory_from, directory_to, end_name=end_name, title=title[method_number],
                                       frame_y=frame, width_param=width_param, height_param=height_param,
                                       legend_column=legend_column, x_label=x_label, y_label=y_label,
                                       method=method_number, grid_x_step=grid_x_step, grid_x_step_min=grid_x_step_min)
