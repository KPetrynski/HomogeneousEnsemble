import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def plot_all_score_methods(name="aver_chunk_score_imb_33_sd_s_rbf_r1_s_rbf_r3",
                           directory_from="results_chunk_size/res_20_sd_s_hyp_r1_s_hyp_r3"):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.set_title(name)
    ax1.set_xlabel('chunk size')
    ax1.set_ylabel('score')
    font_properties = FontProperties()
    font_properties.set_size('small')

    data = np.genfromtxt('%s/%s' % (directory_from, name), delimiter=', ')
    ax1.plot(data[:, 0], data[:, 1], color='r', label='balanced accuracy')
    ax1.plot(data[:, 0], data[:, 2], color='g', label='cohen kappa')
    ax1.plot(data[:, 0], data[:, 3], color='b', label='matthews corrcoef')
    plt.ylim([0, 1])
    ax1.legend(prop=font_properties)
    # plt.show()
    plt.savefig("results_chunk_size_plots/%s" % name)


def plot_compare_streams(names, directory_from, directory_to, title, frame_y, width_param=15, height_param=5,
                         legend_column=2, end_name="", x_label='chunk size', y_label='score', method=1,
                         grid_x_step=500, grid_x_step_min=250):

    fig = plt.figure(figsize=(width_param, height_param), dpi=360)
    ax = fig.add_subplot(111)


    # ax.set_title(title)
    ax.yaxis.label.set_size(25)
    ax.xaxis.label.set_size(25)
    # ax.title.set_size(20)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    max_x = 0

    for name in names:
        data = np.genfromtxt('%s/%s' % (directory_from, name), delimiter=', ')
        ax.plot(data[:, 0], data[:, method], label=name)
        if max_x < data[-1, 0]:
            max_x = data[-1, 0]

    x_major_ticks = np.arange(start=0, stop=max_x+1, step=grid_x_step)
    x_minor_ticks = np.arange(start=0, stop=max_x+1, step=grid_x_step_min)
    y_major_ticks = np.arange(start=frame_y[0], stop=frame_y[1] + 0.1, step=0.1)
    y_minor_ticks = np.arange(start=frame_y[0], stop=frame_y[1] + 0.1, step=0.05)

    ax.set_xticks(x_major_ticks)
    ax.set_xticks(x_minor_ticks, minor=True)
    ax.set_yticks(y_major_ticks)
    ax.set_yticks(y_minor_ticks, minor=True)
    ax.grid(which='minor', alpha=0.5, linestyle="dashed")
    ax.grid(which='major', alpha=1, linestyle="dashed")

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    font_properties = FontProperties()
    font_properties.set_size('xx-large')
    # ax.legend(prop=font_properties)
    ax.legend(loc=9, bbox_to_anchor=(0.5, 1.2), ncol=legend_column, prop=font_properties)

    ax.set_ylim(frame_y)
    ax.set_xlim(0, max_x)
    # plt.ylim(frame_y)
    # plt.xlim([0, max_x])

    art = []
    plt.savefig("%s/%s_%s" % (directory_to, title, end_name), additional_artists=art, bbox_inches="tight")

# plot_all_score_methods()
