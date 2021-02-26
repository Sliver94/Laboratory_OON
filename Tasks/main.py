import pandas as pd
import random as rnd
import numpy as np
import matplotlib.pyplot as plt

from Core.elements import SignalInformation
from Core.elements import Network
from Core.utils import json_path
from Core.science_utils import linear_to_db_conversion
from Core.parameters import Connection

from pathlib import Path
root = Path(__file__).parent.parent


def main():

    # Data frame generation
    df = pd.DataFrame()
    network = Network(root / json_path, df)
    network.connect()
    node_labels = network.nodes.keys()
    pairs = []
    for label1 in node_labels:
        for label2 in node_labels:
            if label1 != label2:
                pairs.append(label1+label2)

    # Columns = ['path', 'latency', 'noise', 'snr']
    paths = []
    latencies = []
    noises = []
    snrs = []
    for pair in pairs:
        for path in network.find_paths(pair[0], pair[1]):
            path_string = ''
            for node in path:
                path_string += node + '->'
            paths.append(path_string[:-2])

            # Propagation
            signal_information = SignalInformation(0.001, path)
            signal_information = network.propagate(signal_information)
            latencies.append(signal_information.latency)
            noises.append(signal_information.noise_power)
            snrs.append(
                linear_to_db_conversion(
                    signal_information.signal_power/signal_information.noise_power
                )
            )
    df['path'] = paths
    df['latency'] = latencies
    df['noise'] = noises
    df['snr'] = snrs
    network.weighted_paths = df

    # Input/Output generation
    input_node = []
    output_node = []
    for i in range(100):

        temp_in = rnd.randint(0, 5)
        while True:
            temp_out = rnd.randint(0, 5)
            if temp_out != temp_in:
                break
        if temp_in == 0:
            input_node.append('A')
        elif temp_in == 1:
            input_node.append('B')
        elif temp_in == 2:
            input_node.append('C')
        elif temp_in == 3:
            input_node.append('D')
        elif temp_in == 4:
            input_node.append('E')
        elif temp_in == 5:
            input_node.append('F')
        if temp_out == 0:
            output_node.append('A')
        elif temp_out == 1:
            output_node.append('B')
        elif temp_out == 2:
            output_node.append('C')
        elif temp_out == 3:
            output_node.append('D')
        elif temp_out == 4:
            output_node.append('E')
        elif temp_out == 5:
            output_node.append('F')

    # Connection generation
    connection_list = []
    for i in range(len(input_node)):
        connection_list.append(Connection(input_node[i], output_node[i], 0.001))


    ##################### FIND BEST SNR
    # Stream call
    network.stream(connection_list, 'snr')
    snr_list = list()
    latency_list = list()

    # Result printing
    print('Best snr case')
    for i in range(len(connection_list)):
        print('For input = ', input_node[i], ', output = ', output_node[i], ', the SNR is: ',
              connection_list[i].snr)
        snr_list.append(connection_list[i].snr)
        latency_list.append(connection_list[i].latency)

    # Removes "none" and "zero" for plotting
    snr_list_no_none = []
    latency_no_zero = []
    for index in range(len(snr_list)):
        if snr_list[index] is not None:
            snr_list_no_none.append(snr_list[index])
        if latency_list[index] != 0:
            latency_no_zero.append(latency_list[index])

    # Conversion to array for plotting
    snr_array = np.array(snr_list_no_none)
    latency_array = np.array(latency_no_zero)

    plt.hist(snr_array, color='blue', edgecolor='black',
             bins=30)
    plt.savefig(root / 'Results/Lab4/snr_distribution_find_best_snr')
    plt.show()

    plt.hist(latency_array, color='blue', edgecolor='black',
             bins=30)
    plt.savefig(root / 'Results/Lab4/latency_distribution_find_best_snr')
    plt.show()

"""
    ##################### FIND BEST LATENCY
    # Stream call
    network.stream(connection_list, 'latency')
    snr_list = list()
    latency_list = list()

    # Result printing
    print('Best latency case')
    for i in range(len(connection_list)):
        print('For input = ', input_node[i], ', output = ', output_node[i], ', the SNR is: ',
              connection_list[i].snr)
        snr_list.append(connection_list[i].snr)
        latency_list.append(connection_list[i].latency)

    # Removes "none" and "zero" for plotting
    snr_list_no_none = []
    latency_no_zero = []
    for index in range(len(snr_list)):
        if snr_list[index] is not None:
            snr_list_no_none.append(snr_list[index])
        if latency_list[index] != 0:
            latency_no_zero.append(latency_list[index])

    # Conversion to array for plotting
    snr_array = np.array(snr_list_no_none)
    latency_array = np.array(latency_no_zero)

    plt.hist(snr_array, color='blue', edgecolor='black',
             bins=30)
    plt.savefig(root / 'Results/Lab4/snr_distribution_find_best_latency')
    plt.show()

    plt.hist(latency_array, color='blue', edgecolor='black',
             bins=30)
    plt.savefig(root / 'Results/Lab4/latency_distribution_find_best_latency')
    plt.show()
"""

if __name__ == '__main__':
    main()
