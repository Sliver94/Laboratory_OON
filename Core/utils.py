import random as rnd
import numpy as np
import matplotlib.pyplot as plt

c = 3 * (10 ** 8)                # Speed of light
c_network = 2/3 * c              # Speed of signal on the network
json_path1 = 'Resources/nodes_full.json'  # Json file address
json_path2 = 'Resources/nodes_not_full.json'  # Json file address
snr_or_latency_choice = 'snr'
# snr_or_latency_choice = 'latency'
number_of_connections = 100
input_signal_power = 0.001


def generate_requests():
    input_node = []
    output_node = []
    for i in range(number_of_connections):
        temp_in = rnd.randint(0, 5)
        while True:
            temp_out = rnd.randint(0, 5)
            if temp_out != temp_in:
                break
        number_to_node = ['A', 'B', 'C', 'D', 'E', 'F']
        input_node.append(number_to_node[temp_in])
        output_node.append(number_to_node[temp_out])
    return [input_node, output_node]


def generate_output_arrays(connection_list):
    snr_list = list()
    latency_list = list()

    for i in range(len(connection_list)):
        snr_list.append(connection_list[i].snr)
        latency_list.append(connection_list[i].latency)

    number_of_blocks = 0
    snr_list_no_none = []
    latency_no_zero = []
    for index in range(len(snr_list)):
        if snr_list[index] is not None:
            snr_list_no_none.append(snr_list[index])
        else:
            number_of_blocks = number_of_blocks + 1
        if latency_list[index] != 0:
            latency_no_zero.append(latency_list[index])

    # Conversion to array for plotting
    snr_array = np.array(snr_list_no_none)
    latency_array = np.array(latency_no_zero)
    return [snr_array, latency_array, number_of_blocks]


def plot_results(root, snr_array, latency_array, snr_array2, latency_array2, number_of_blocks_full,
                 number_of_blocks_not_full):
    plt.hist(snr_array, color='blue', edgecolor='black', bins=50)
    plt.xlabel('SNR [dB]')
    plt.ylabel('Frequency')
    plt.title('SNR Distribution (Full Switching Matrices)')
    plt.savefig(root / 'Results/Lab7/snr_distribution_full_sw_mx.png')
    plt.show()

    plt.hist(latency_array*1e3, color='red', edgecolor='black', bins=50)
    plt.xlabel('Latency [ms]')
    plt.ylabel('Frequency')
    plt.title('Latency Distribution (Full Switching Matrices)')
    plt.savefig(root / 'Results/Lab7/latency_distribution_full_sw_mx.png')
    plt.show()

    plt.hist(snr_array2, color='blue', edgecolor='black', bins=50)
    plt.xlabel('SNR [dB]')
    plt.ylabel('Frequency')
    plt.title('SNR Distribution (Not Full Switching Matrices)')
    plt.savefig(root / 'Results/Lab7/snr_distribution_not_full_sw_mx.png')
    plt.show()

    plt.hist(latency_array2*1e3, color='red', edgecolor='black', bins=50)
    plt.xlabel('Latency [ms]')
    plt.ylabel('Frequency')
    plt.title('Latency Distribution (Not Full Switching Matrices)')
    plt.savefig(root / 'Results/Lab7/latency_distribution_not_full_sw_mx.png')
    plt.show()

    print('Blocking events for full switching matrix = ', number_of_blocks_full)
    print('Blocking events for not full switching matrix = ', number_of_blocks_not_full)
    return
