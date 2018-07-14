import plot_from_csv
from os import listdir
from os.path import isfile, join


def get_files_names(directory):
    data_set_names = [f for f in listdir("%s/" % directory) if isfile(join("%s/" % directory, f))]
    print(data_set_names)
    return data_set_names


# ------------------ --average
# frame = [0.5, 1]
# width_param = 10
# height_param = 10
# legend_column = 2
# grid_x_step = 500
# grid_x_step_min = 250

# --------------------- stream
frame = [0.4, 1]
width_param = 20
height_param = 10
legend_column = 3
grid_x_step = 10000
grid_x_step_min = 1000

sub_directory = "hyp"
end_name = "hyp_zoom"
directory_from = "results_chunk_size/" + sub_directory
directory_to = "results_chunk_size_plots/" + sub_directory
title = ["", "average balanced accuracy", "average cohen kappa", "average matthews corrcoef"]
score_method_name = ["", "balanced accuracy", "cohen kappa", "matthews corrcoef"]
x_label = "data stream"
# y_label = score_method_name[method_number] + " score"
names = get_files_names(directory_from)
# names = ["aver_chunk_score_imb_20_sd_s_rbf_r1_s_rbf_r3"]
for method_number in range(1, 2):
    y_label = score_method_name[method_number] + " score"
    plot_from_csv.plot_compare_streams(names, directory_from, directory_to, end_name=end_name, title=title[method_number],
                                       frame_y=frame, width_param=width_param, height_param=height_param,
                                       legend_column=legend_column, x_label=x_label, y_label=y_label,
                                       method=method_number, grid_x_step=grid_x_step, grid_x_step_min=grid_x_step_min)
