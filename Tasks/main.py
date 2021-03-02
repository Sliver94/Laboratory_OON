import random as rnd
import numpy as np
import matplotlib.pyplot as plt

from Core.elements import Network
from Core.utils import json_path1
from Core.utils import json_path2
from Core.utils import input_signal_power
from Core.utils import number_of_connections
from Core.utils import snr_or_latency_choice
from Core.parameters import Connection

from pathlib import Path
root = Path(__file__).parent.parent


def main():
    # Initialize an object network of class Network
    network = Network(root / json_path1)

    # Fills "successive" attributes of Nodes and Lines
    network.connect()

    # Fills weighted paths and initialize route_space attributes
    network.initialize(root / json_path1)

    # Input/Output generation
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

    # Connection generation
    connection_list = []
    for i in range(len(input_node)):
        connection_list.append(Connection(input_node[i], output_node[i], input_signal_power))

    # Stream call
    network.stream(connection_list, snr_or_latency_choice)
    snr_list = list()
    latency_list = list()

    # Results generation
    for i in range(len(input_node)):
        snr_list.append(connection_list[i].snr)
        latency_list.append(connection_list[i].latency)

    number_of_blocks_full = 0
    snr_list_no_none = []
    latency_no_zero = []
    for index in range(len(snr_list)):
        if snr_list[index] is not None:
            snr_list_no_none.append(snr_list[index])
        else:
            number_of_blocks_full = number_of_blocks_full + 1
        if latency_list[index] != 0:
            latency_no_zero.append(latency_list[index])

    # Conversion to array for plotting
    snr_array = np.array(snr_list_no_none)
    latency_array = np.array(latency_no_zero)

    # Result plotting
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

    # Initialize an object network of class Network
    network2 = Network(root / json_path2)

    # Fills "successive" attributes of Nodes and Lines
    network2.connect()

    # Fills weighted paths and initialize route_space attributes
    network2.initialize(root / json_path2)

    # Connection generation
    connection_list2 = []
    for i in range(len(input_node)):
        connection_list2.append(Connection(input_node[i], output_node[i], input_signal_power))

    # Stream call
    network2.stream(connection_list2, snr_or_latency_choice)
    snr_list2 = list()
    latency_list2 = list()

    # Results generation
    for i in range(len(input_node)):
        snr_list2.append(connection_list2[i].snr)
        latency_list2.append(connection_list2[i].latency)

    number_of_blocks_not_full = 0
    snr_list_no_none2 = []
    latency_no_zero2 = []
    for index in range(len(snr_list2)):
        if snr_list2[index] is not None:
            snr_list_no_none2.append(snr_list2[index])
        else:
            number_of_blocks_not_full = number_of_blocks_not_full + 1
        if latency_list2[index] != 0:
            latency_no_zero2.append(latency_list2[index])

    # Conversion to array for plotting
    snr_array2 = np.array(snr_list_no_none2)
    latency_array2 = np.array(latency_no_zero2)

    # Result plotting
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


if __name__ == '__main__':
    main()
