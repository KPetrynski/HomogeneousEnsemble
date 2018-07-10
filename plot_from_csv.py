import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def plot_all_score_methods(name="aver_chunk_score_imb_33_sd_s_rbf_r1_s_rbf_r3",
                           directory_from="results_chunk_size/average_score"):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.set_title(name)
    ax1.set_xlabel('chunk size')
    ax1.set_ylabel('score')

    data = np.genfromtxt('%s/%s' % (directory_from, name), delimiter=', ')
    ax1.plot(data[:, 0], data[:, 1], color='r', label='balanced accuracy')
    ax1.plot(data[:, 0], data[:, 2], color='g', label='cohen kappa')
    ax1.plot(data[:, 0], data[:, 3], color='b', label='matthews corrcoef')
    plt.ylim([0, 1])
    ax1.legend()
    # plt.show()
    plt.savefig("results_chunk_size_plots/%s" % name)


def plot_compare_streams(names, directory_from, title, x_label='chunk size', y_label='score', method=1):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.set_title(title)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    font_properties = FontProperties()
    font_properties.set_size('small')

    for name in names:
        data = np.genfromtxt('%s/%s' % (directory_from, name), delimiter=', ')
        ax1.plot(data[:, 0], data[:, method], label=name)
    ax1.legend(prop=font_properties)
    plt.ylim([0.65, 0.85])
    plt.savefig("results_smoke_weights_plots/%s_%s" % (title, "av_zoom"))


plot_all_score_methods()
