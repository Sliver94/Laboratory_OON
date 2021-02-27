import random as rnd
import numpy as np
import matplotlib.pyplot as plt

from Core.elements import Network
from Core.utils import json_path1
from Core.utils import json_path2
from Core.utils import json_path3
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
    bit_rate_list = list()

    # Result printing
    for i in range(len(input_node)):
        snr_list.append(connection_list[i].snr)
        latency_list.append(connection_list[i].latency)
        bit_rate_list.append(connection_list[i].bit_rate)

    number_of_blocks_full_fixed = 0
    snr_list_no_none = []
    latency_no_zero = []
    bit_rate_no_zero = []
    for index in range(len(snr_list)):
        if snr_list[index] is not None:
            snr_list_no_none.append(snr_list[index])
        else:
            number_of_blocks_full_fixed = number_of_blocks_full_fixed + 1
        if latency_list[index] != 0:
            latency_no_zero.append(latency_list[index])
        if bit_rate_list[index] != 0:
            bit_rate_no_zero.append(bit_rate_list[index])

    # Conversion to array for plotting
    snr_array = np.array(snr_list_no_none)
    latency_array = np.array(latency_no_zero)
    bit_rate_array = np.array(bit_rate_no_zero)

    # 2nd NETWORK ################################################################

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
    bit_rate_list2 = list()

    # Result printing
    for i in range(len(input_node)):
        snr_list2.append(connection_list2[i].snr)
        latency_list2.append(connection_list2[i].latency)
        bit_rate_list2.append(connection_list2[i].bit_rate)

    number_of_blocks_full_flex = 0
    snr_list_no_none2 = []
    latency_no_zero2 = []
    bit_rate_no_zero2 = []

    for index in range(len(snr_list2)):
        if snr_list2[index] is not None:
            snr_list_no_none2.append(snr_list2[index])
        else:
            number_of_blocks_full_flex = number_of_blocks_full_flex + 1
        if latency_list2[index] != 0:
            latency_no_zero2.append(latency_list2[index])
        if bit_rate_list2[index] != 0:
            bit_rate_no_zero2.append(bit_rate_list2[index])

    # Conversion to array for plotting
    snr_array2 = np.array(snr_list_no_none2)
    latency_array2 = np.array(latency_no_zero2)
    bit_rate_array2 = np.array(bit_rate_no_zero2)

    # 3rd NETWORK ###################################################################

    # Initialize an object network of class Network
    network3 = Network(root / json_path3)

    # Fills "successive" attributes of Nodes and Lines
    network3.connect()

    # Fills weighted paths and initialize route_space attributes
    network3.initialize(root / json_path3)

    # Connection generation
    connection_list3 = []
    for i in range(len(input_node)):
        connection_list3.append(Connection(input_node[i], output_node[i], input_signal_power))

    # Stream call
    network3.stream(connection_list3, snr_or_latency_choice)
    snr_list3 = list()
    latency_list3 = list()
    bit_rate_list3 = list()

    # Result printing
    for i in range(len(input_node)):
        snr_list3.append(connection_list3[i].snr)
        latency_list3.append(connection_list3[i].latency)
        bit_rate_list3.append(connection_list3[i].bit_rate)

    number_of_blocks_full_shannon = 0
    snr_list_no_none3 = []
    latency_no_zero3 = []
    bit_rate_no_zero3 = []
    for index in range(len(snr_list3)):
        if snr_list3[index] is not None:
            snr_list_no_none3.append(snr_list3[index])
        else:
            number_of_blocks_full_shannon = number_of_blocks_full_shannon + 1
        if latency_list3[index] != 0:
            latency_no_zero3.append(latency_list3[index])
        if bit_rate_list3[index] != 0:
            bit_rate_no_zero3.append(bit_rate_list3[index])

    # Conversion to array for plotting
    snr_array3 = np.array(snr_list_no_none3)
    latency_array3 = np.array(latency_no_zero3)
    bit_rate_array3 = np.array(bit_rate_no_zero3)

    print('Blocking events for full fixed rate = ', number_of_blocks_full_fixed)
    print('Blocking events for full flex rate = ', number_of_blocks_full_flex)
    print('Blocking events for full shannon = ', number_of_blocks_full_shannon)

    file = open(root / 'Results/Lab9/bit_rates_and_capacities.txt', "a")

    bit_rate_mean = np.mean(bit_rate_array)
    bit_rate_mean2 = np.mean(bit_rate_array2)
    bit_rate_mean3 = np.mean(bit_rate_array3)
    print('Bit rate mean for full fixed rate = ', bit_rate_mean, 'Gbps')
    file.write('Bit rate mean for full fixed rate = ' + str(bit_rate_mean) + ' Gbps\n')
    print('Bit rate mean for full flex rate = ', bit_rate_mean2, 'Gbps')
    file.write('Bit rate mean for full flex rate = ' + str(bit_rate_mean2) + ' Gbps\n')
    print('Bit rate mean for full shannon = ', bit_rate_mean3, 'Gbps')
    file.write('Bit rate mean for full shannon = ' + str(bit_rate_mean3) + ' Gbps\n\n')

    capacity = np.sum(bit_rate_array)
    capacity2 = np.sum(bit_rate_array2)
    capacity3 = np.sum(bit_rate_array3)
    print('Capacity for full fixed rate = ', capacity, 'Gbps')
    file.write('Capacity for full fixed rate = ' + str(capacity) + ' Gbps\n')
    print('Capacity mean for full flex rate = ', capacity2, 'Gbps')
    file.write('Capacity for full flex rate = ' + str(capacity2) + ' Gbps\n')
    print('Capacity mean for full shannon = ', capacity3, 'Gbps')
    file.write('Capacity for full shannon = ' + str(capacity3) + ' Gbps\n')
    file.close()

    # Result plotting

    plt.hist(snr_array, color='blue', edgecolor='black', bins=50)
    plt.savefig(root / 'Results/Lab9/snr_distribution_fixed.png')
    plt.show()
    plt.hist(latency_array, color='blue', edgecolor='black', bins=50)
    plt.savefig(root / 'Results/Lab9/latency_distribution_fixed.png')
    plt.show()
    plt.hist(bit_rate_array, color='blue', edgecolor='black', bins=50)
    plt.savefig(root / 'Results/Lab9/bit_rate_distribution_fixed.png')
    plt.show()

    plt.hist(snr_array2, color='blue', edgecolor='black', bins=50)
    plt.savefig(root / 'Results/Lab9/snr_distribution_flex.png')
    plt.show()
    plt.hist(latency_array2, color='blue', edgecolor='black', bins=50)
    plt.savefig(root / 'Results/Lab9/latency_distribution_flex.png')
    plt.show()
    plt.hist(bit_rate_array2, color='blue', edgecolor='black', bins=50)
    plt.savefig(root / 'Results/Lab9/bit_rate_distribution_flex.png')
    plt.show()

    plt.hist(snr_array3, color='blue', edgecolor='black', bins=50)
    plt.savefig(root / 'Results/Lab9/snr_distribution_shannon.png')
    plt.show()
    plt.hist(latency_array3, color='blue', edgecolor='black', bins=50)
    plt.savefig(root / 'Results/Lab9/latency_distribution_shannon.png')
    plt.show()
    plt.hist(bit_rate_array3, color='blue', edgecolor='black', bins=50)
    plt.savefig(root / 'Results/Lab9/bit_rate_distribution_shannon.png')
    plt.show()


if __name__ == '__main__':
    main()
