import matplotlib
import matplotlib.pyplot as plt
import numpy as np

c = 3 * (10 ** 8)  # Speed of light
c_network = 2 / 3 * c  # Speed of signal on the network
json_path1 = 'Resources/nodes_full_fixed_rate.json'  # Json file address
json_path2 = 'Resources/nodes_full_flex_rate.json'  # Json file address
json_path3 = 'Resources/nodes_full_shannon.json'  # Json file address
snr_or_latency_choice = 'snr'
# snr_or_latency_choice = 'latency'
number_of_connections = 100
input_signal_power = 0.001
number_of_channels = 10
gain = 16  # dB
noise_figure = 3  # dB
h = 6.62607015e-34  # Plack constant  J * s
f = 193.414e12  # C-band center frequency
alpha = 0.2  # dB/km
beta2 = 2.13e-26  # (m Hz^2)^-1
gamma = 1.27e-3  # (m W)^-1
# f_max = 191.2e12
# f_min = 195.6e12
# df = (f_max - f_min) / number_of_channels
df = 50e9
Rs = 32e9
Bn = 12.5e9
BERt = 10e-3


def print_input_matrix(matrix, node_list, root, path):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap='winter')

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(node_list)))
    ax.set_yticks(np.arange(len(node_list)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(node_list)
    ax.set_yticklabels(node_list)

    # Rotate the tick labels and set their alignment.
#    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(node_list)):
        for j in range(len(node_list)):
            text = ax.text(j, i, matrix[i, j],
                           ha="center", va="center", color="k")

    ax.set_title("Input Traffic Matrix")
    fig.tight_layout()
    plt.savefig(root / path)
    plt.show()
    return


def print_output_matrix(matrix, node_list, root, path):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap='winter')

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(node_list)))
    ax.set_yticks(np.arange(len(node_list)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(node_list)
    ax.set_yticklabels(node_list)

    # Rotate the tick labels and set their alignment.
#    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(node_list)):
        for j in range(len(node_list)):
            text = ax.text(j, i, matrix[i, j],
                           ha="center", va="center", color="k")

    ax.set_title("Output Traffic Matrix")
    fig.tight_layout()
    plt.savefig(root / path)
    plt.show()
    return
