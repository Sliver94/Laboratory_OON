import random as rnd
import numpy as np
import matplotlib.pyplot as plt

from Core.elements import Network
from Core.utils import json_path
from Core.utils import snr_or_latency_choice
from Core.parameters import Connection

from pathlib import Path
root = Path(__file__).parent.parent


def main():

    # Initialize an object network of class Network
    network = Network(root / json_path)

    # Fills "successive" attributes of Nodes and Lines
    network.connect()

    # Fills weighted paths and initialize route_space attributes
    network.initialize()

    # Input/Output generation
    input_signal_power = 0.001
    input_node = []
    output_node = []
    for i in range(100):
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

    # Results printing
    print('Best', snr_or_latency_choice, 'case:')
    for i in range(len(input_node)):
        print('Connection number:', i)
        print('Input node =', input_node[i], ', output node =', output_node[i])
        print('SNR =', connection_list[i].snr, ', Latency =', connection_list[i].latency)
        snr_list.append(connection_list[i].snr)
        latency_list.append(connection_list[i].latency)

    # Removes "None" and "0"
    snr_list_no_none = []
    latency_no_zero = []
    for index in range(len(snr_list)):
        if snr_list[index] is not None:
            snr_list_no_none.append(snr_list[index])
        if latency_list[index] != 0:
            latency_no_zero.append(latency_list[index])

    snr_array = np.array(snr_list_no_none)
    latency_array = np.array(latency_no_zero)

    plt.hist(snr_array, color='blue', edgecolor='black', bins=50)
    plt.savefig(root / 'Results/Lab5/snr_distribution_find_best_snr')
    plt.show()

    plt.hist(latency_array, color='blue', edgecolor='black', bins=50)
    plt.savefig(root / 'Results/Lab5/latency_distribution_find_best_snr')
    plt.show()


if __name__ == '__main__':
    main()
